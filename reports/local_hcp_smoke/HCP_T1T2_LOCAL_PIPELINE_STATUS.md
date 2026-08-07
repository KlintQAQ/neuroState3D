# Local HCP T1/T2 Pipeline Status

Status: `LOCAL REAL HCP T1/T2 PIPELINE: PENDING USER AUTHENTICATION`

This repository now contains the local validation infrastructure for
NeuroState-3D on real HCP Young Adult structural MRI, but the local machine
does not currently contain authorized HCP T1w/T2w NIfTI files. The code
therefore stops before claiming a real-HCP forward pass.

## Data Boundary

- Use only official HCP Young Adult 2025 structural MRI data obtained through
  authorized HCP/ConnectomeDB access.
- Do not use mirrors, Kaggle/HuggingFace copies, DWI, fMRI, ADNI, BraTS, EEG,
  or PET data for this local smoke validation.
- Keep source code in this repository and local data under
  `D:\NeuroStateData\HCP_YA_2025`.

## Expected Local Layout

```text
D:\NeuroStateData\HCP_YA_2025\
  raw\
  manifests\
  processed\
  cache\
```

Populate `manifests\hcp_local_smoke.json` after authorized download. The
manifest `data_root` should be `D:\NeuroStateData\HCP_YA_2025`, and modality
paths should be relative paths such as `raw\...\T1w.nii.gz`.

## Commands

Inspect one subject:

```bash
python scripts/inspect_hcp_subject.py \
  --manifest manifests/hcp_local_smoke.json \
  --preprocess \
  --output outputs/hcp_t1_t2_inspection.json
```

Run real HCP T1/T2 forward smoke:

```bash
python scripts/test_real_hcp.py \
  --config configs/hcp.yaml \
  --device cuda
```

Run synthetic NIfTI tests:

```bash
python -m unittest tests.test_hcp_dataset
```

## Validation Contract

- Modality order: `[t1, t2, fa, md, alff]`
- Present structural modalities: T1/T2
- Missing modalities: FA/MD/ALFF as `None`
- Modality mask: `[1, 1, 0, 0, 0]`
- Fusion is blocked if T1/T2 are not in the same physical space.
