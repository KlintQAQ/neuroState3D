from dataclasses import asdict
from types import SimpleNamespace
import csv
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.slice_drift_transport_generator import (
    SliceDriftTransportGenerator, SliceDriftTransportGeneratorConfig,
)
from scripts.train_slice_virtual_modality_drifting import (
    evaluate, exact_resume_config_mismatches, freeze_prompted_base,
    selection_score, closed_loop_no_harm_loss,
    BraTSSliceDataset,
)
from scripts.visualize_slice_virtual_modality_generation import load_model
from utils.brats_metrics import segmentation_loss


def small_model():
    return SliceDriftTransportGenerator(SliceDriftTransportGeneratorConfig(
        hidden_channels=8, transport_steps=2, closed_loop_token_drift=True,
        closed_loop_feedback_channels=4, closed_loop_token_channels=8,
        closed_loop_token_heads=2,
    ))


def inputs():
    return torch.randn(2, 4, 16, 16), torch.tensor([[1., 0., 1., 1.]]).repeat(2, 1)


def test_weighted_segmentation_loss_and_gradient_ignore_batch_replication():
    torch.manual_seed(4)
    x = torch.randn(1, 3, 12, 12, requires_grad=True)
    y = (torch.rand_like(x) > .6).float()
    loss = segmentation_loss(x, y)
    grad, = torch.autograd.grad(loss, x)
    repeated = x.detach().repeat(5, 1, 1, 1).requires_grad_()
    repeated_loss = segmentation_loss(repeated, y.repeat(5, 1, 1, 1))
    repeated_grad, = torch.autograd.grad(repeated_loss, repeated)
    torch.testing.assert_close(loss, repeated_loss)
    torch.testing.assert_close(grad, repeated_grad.sum(0, keepdim=True))
    assert segmentation_loss(repeated, y.repeat(5, 1, 1, 1), legacy_batch_sum=True) > loss


def test_reference_can_be_skipped_without_changing_outputs_or_gradients():
    torch.manual_seed(5)
    model = small_model()
    with torch.no_grad():
        model.token_drift_adapter.delta_head.weight.normal_(std=.1)
    x, mask = inputs()
    outputs, gradients = [], []
    for reference in (True, False):
        model.zero_grad(set_to_none=True)
        result = model(x, mask, compute_reference=reference)
        result['synthetic'].square().mean().backward()
        outputs.append(result)
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert 'closed_loop_reference_synthetic' not in outputs[1]
    for key in outputs[1]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    assert gradients[0].keys() == gradients[1].keys()
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=0, atol=0)


def test_frozen_base_remains_fixed_while_adapter_learns():
    model = small_model()
    freeze_prompted_base(model)
    old = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    x, mask = inputs()
    before = model(x, mask)
    (before['synthetic'] - .7).square().mean().backward()
    optimizer.step()
    after = model(x, mask)
    for n, p in model.named_parameters():
        if n in old:
            torch.testing.assert_close(old[n], p, rtol=0, atol=0)
    torch.testing.assert_close(before['closed_loop_reference_synthetic'],
                               after['closed_loop_reference_synthetic'], rtol=0, atol=0)
    assert not torch.equal(before['synthetic'], after['synthetic'])


def test_legacy_resume_requires_explicit_legacy_objective_and_metrics():
    old = {'batch_size': 4}
    new = {**old, 'closed_loop_token_drift': False, 'closed_loop_adapter_scale': .1,
           'segmentation_loss_version': 1, 'evaluation_version': 1}
    assert exact_resume_config_mismatches(old, new) == {}
    new['segmentation_loss_version'] = 2
    assert set(exact_resume_config_mismatches(old, new)) == {'segmentation_loss_version'}
    new['closed_loop_token_drift'] = True
    assert 'closed_loop_token_drift' in exact_resume_config_mismatches(old, new)


def test_mmap_dataset_matches_eager_loading_across_epochs(tmp_path):
    rng = np.random.default_rng(9)
    image = rng.normal(size=(4, 16, 16, 9))
    seg = rng.integers(0, 4, size=(16, 16, 9), dtype=np.int16)
    np.save(tmp_path / 'image.npy', image)
    np.save(tmp_path / 'seg.npy', seg)
    manifest = tmp_path / 'manifest.csv'
    row = {'dataset': 'GLI', 'subject_id': 'example', 'status': 'processed',
           'multimodal_path': str(tmp_path / 'image.npy'), 'seg_path': str(tmp_path / 'seg.npy')}
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    dataset = BraTSSliceDataset(manifest, spatial_size=12, slices_per_subject=2,
                               max_subjects=None, seed=46,
                               target_modality='t1c', slice_context_radius=1,
                               slice_crop_size=10, slice_crop_mode='region_balanced')
    original_load = np.load

    def eager(path, **kwargs):
        value = original_load(path)
        return value.astype(np.float32 if value.ndim == 4 else np.int64)

    for epoch in (0, 2):
        dataset.set_epoch(epoch)
        actual = dataset[1]
        with patch('scripts.train_slice_virtual_modality_drifting.np.load', side_effect=eager):
            expected = dataset[1]
        for key in expected:
            if torch.is_tensor(expected[key]):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            else:
                assert actual[key] == expected[key]


class MetricModel(torch.nn.Module):
    config = SimpleNamespace(class_conditioned=False)

    def forward(self, image, mask, condition=None):
        synthetic = image[:, :1] * .6
        return {'synthetic': synthetic, 'uncertainty': synthetic.abs() + .1,
                'confidence': torch.exp(-synthetic.abs() - .1),
                'closed_loop_reference_synthetic': synthetic + .03}


def test_slice_validation_is_independent_of_batch_partition():
    torch.manual_seed(6)
    samples = [{'image': torch.rand(4, 12, 12), 'target_index': 1,
                'observed_mask': torch.tensor([1., 0., 1., 1.]),
                'target_regions': (torch.rand(3, 12, 12) > .5).float()}
               for _ in range(5)]
    a = evaluate(MetricModel(), DataLoader(samples, batch_size=1), 'cpu')
    b = evaluate(MetricModel(), DataLoader(samples, batch_size=3), 'cpu')
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-7, err_msg=key)


def test_closed_loop_selection_penalizes_reference_regression():
    base = {'mae': .1, 'closed_loop_reference_mae': .1,
            'closed_loop_mae_delta_from_reference': 0., 'closed_loop_harm_rate': 0.}
    worse = {**base, 'closed_loop_reference_mae': .05,
             'closed_loop_mae_delta_from_reference': .05, 'closed_loop_harm_rate': .8}
    assert selection_score(worse, 'closed_loop_noharm_composite') > selection_score(
        base, 'closed_loop_noharm_composite')


def test_noharm_lesion_mask_excludes_background_exactly():
    target = torch.zeros(1, 1, 4, 4)
    regions = torch.zeros(1, 3, 4, 4)
    regions[:, :, 0, 0] = 1
    synthetic = torch.ones_like(target)
    synthetic[:, :, 0, 0] = 0
    value = closed_loop_no_harm_loss(
        {'synthetic': synthetic, 'closed_loop_reference_synthetic': target},
        target, regions, 0, 0., 'lesion')
    assert value.item() == 0


def test_inference_loads_legacy_unused_head_but_rejects_missing_active_weights(tmp_path):
    model = small_model()
    state = dict(model.state_dict())
    state['feedback_controller.failure_head.weight'] = torch.randn(1, 8, 1, 1)
    state['feedback_controller.failure_head.bias'] = torch.zeros(1)
    config = {**asdict(model.config), 'model_kind': 'transport', 'slice_context_radius': 0}
    checkpoint = tmp_path / 'model.pt'
    torch.save({'model': state, 'config': config, 'target_modality': 't1c'}, checkpoint)
    loaded, _ = load_model(str(checkpoint), 'cpu', '')
    x, mask = inputs()
    torch.testing.assert_close(model(x, mask)['synthetic'], loaded(x, mask)['synthetic'])
    del state['token_drift_adapter.delta_head.weight']
    torch.save({'model': state, 'config': config, 'target_modality': 't1c'}, checkpoint)
    try:
        load_model(str(checkpoint), 'cpu', '')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Incomplete checkpoint was accepted')
