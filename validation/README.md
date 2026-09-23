# External validation

This directory contains reproducible external validation code. Downloaded
medical images and generated results are intentionally ignored by Git.

## SynthRAD2023 pelvis CBCT smoke test

The validation uses nine paired pelvis CBCT/CT cases from the three contributing
centres in SynthRAD2023 Task 2. The dataset is licensed CC BY-NC 4.0.

```powershell
uv sync --group validation
uv run --group validation python validation/download_synthrad.py
uv run --group validation python validation/validate_synthrad.py
```

The dataset does not provide manual bone labels. The paired, rigidly registered
CT is segmented with a fixed 150/300 HU hysteresis configuration and used only
as a cross-modality proxy. Reported overlap therefore measures CBCT/CT
stability, not clinical accuracy.
The validation also applies the dataset's patient-outline mask and aligns paired
volumes from their canonical LPS geometry before computing overlap and surface
metrics.

Dataset citation:

> Thummerer A, et al. SynthRAD2023 Grand Challenge dataset: Generating
> synthetic CT for radiotherapy. Medical Physics. 2023;50(7):4664-4674.
> https://doi.org/10.1002/mp.16529
