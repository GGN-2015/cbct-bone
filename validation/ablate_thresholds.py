"""Compare anatomy-independent bone-threshold percentiles on SynthRAD2023."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.filters import threshold_multiotsu
from validate_synthrad import _dice, _resample_to_reference

from cbct_bone import (
    AdaptiveBoneSegmenter,
    BoneSegmentationConfig,
    MedicalVolumeReader,
)

DATA_ROOT = Path(__file__).parents[1] / ".validation-data" / "synthrad2023"
WEAK_PERCENTILES = (90.0, 92.0, 94.0, 96.0)
STRONG_PERCENTILE = 98.5


def _body_intensities(volume: np.ndarray) -> np.ndarray:
    values = volume[np.isfinite(volume)]
    stride = max(1, values.size // 1_000_000)
    sample = values[::stride]
    low, high = np.percentile(sample, (0.1, 99.9))
    sample = sample[(sample >= low) & (sample <= high)]
    air_threshold = threshold_multiotsu(sample, classes=3, nbins=256)[0]
    return sample[sample > air_threshold]


def main() -> None:
    reader = MedicalVolumeReader(spacing_mm=1.0)
    for case_dir in sorted(DATA_ROOT.iterdir()):
        cbct = reader.read(case_dir / "cbct.nii.gz")
        ct = reader.read(case_dir / "ct.nii.gz")
        outline = reader.read(case_dir / "mask.nii.gz")
        ct_reference, _ = AdaptiveBoneSegmenter().segment(ct.data, ct.spacing_mm)
        ct_reference = _resample_to_reference(ct_reference, ct, cbct, order=0).astype(
            bool
        )
        support = _resample_to_reference(outline.data, outline, cbct, order=0) > 0.5
        ct_reference &= support
        body = _body_intensities(cbct.data)
        rows: list[tuple[float, ...]] = []
        for percentile in WEAK_PERCENTILES:
            weak, strong = np.percentile(body, (percentile, STRONG_PERCENTILE))
            segmenter = AdaptiveBoneSegmenter(
                BoneSegmentationConfig(
                    weak_threshold=float(weak),
                    strong_threshold=float(strong),
                )
            )
            prediction, _ = segmenter.segment(cbct.data, cbct.spacing_mm)
            prediction = prediction.astype(bool) & support
            dice, sensitivity, precision = _dice(prediction, ct_reference)
            rows.append(
                (
                    percentile,
                    round(float(weak), 1),
                    round(float(strong), 1),
                    round(dice, 3),
                    round(sensitivity, 3),
                    round(precision, 3),
                    round(prediction.sum() / max(1, ct_reference.sum()), 2),
                )
            )
        print(case_dir.name, rows, flush=True)


if __name__ == "__main__":
    main()
