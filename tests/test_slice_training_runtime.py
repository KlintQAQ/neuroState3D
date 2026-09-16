import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from scripts.train_slice_virtual_modality_drifting import (
    ResumableBatchSampler,
    exact_resume_config_mismatches,
    sanitize_gradients,
)


def test_sanitize_preserves_finite_values_and_counts_all_nonfinite_values():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    parameters = list(model.parameters())
    parameters[0].grad = torch.tensor([[1.0, float("nan"), float("inf")],
                                       [float("-inf"), -2.0, 0.0]])
    parameters[1].grad = torch.tensor([3.0, 4.0])
    original_finite_grad = parameters[1].grad
    assert sanitize_gradients(model) == 3
    assert torch.equal(parameters[0].grad, torch.tensor([[1., 0., 0.], [0., -2., 0.]]))
    assert parameters[1].grad is original_finite_grad
    assert parameters[2].grad is None
    assert sanitize_gradients(model) == 0
    assert sanitize_gradients(torch.nn.Linear(1, 1)) == 0


def test_resume_sampler_preserves_order_and_rng_without_loading_skipped_samples():
    class CountingDataset(Dataset):
        def __init__(self):
            self.loaded = []

        def __len__(self):
            return 23

        def __getitem__(self, index):
            self.loaded.append(index)
            return index

    baseline_data = CountingDataset()
    baseline_rng = torch.Generator().manual_seed(4601)
    baseline = list(DataLoader(baseline_data, batch_size=4, shuffle=True,
                               generator=baseline_rng))
    resumed_data = CountingDataset()
    resumed_rng = torch.Generator().manual_seed(4601)
    sampler = ResumableBatchSampler(RandomSampler(resumed_data, generator=resumed_rng),
                                    batch_size=4, drop_last=False)
    sampler.start_batch = 3
    resumed = list(DataLoader(resumed_data, batch_sampler=sampler, generator=resumed_rng))
    assert len(resumed) == len(baseline) - 3
    assert all(torch.equal(a, b) for a, b in zip(baseline[3:], resumed))
    assert len(resumed_data.loaded) == 11
    assert torch.equal(baseline_rng.get_state(), resumed_rng.get_state())
    sampler.start_batch = 0
    assert len(sampler) == len(baseline)


def test_runtime_options_do_not_reset_checkpoint_training_state():
    old = {"batch_size": 32, "lr": 5e-5}
    new = {**old, "cpu_threads": 4, "profile_steps": 5, "num_workers": 12}
    assert exact_resume_config_mismatches(old, new) == {}
    new["lr"] = 1e-4
    assert set(exact_resume_config_mismatches(old, new)) == {"lr"}
