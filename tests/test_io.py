from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ct_mri_dicom_nii_reader import BodyData

from cbct_bone.io import MedicalVolumeReader, load_volume, save_mask


def test_nifti_mask_round_trip_through_required_reader(tmp_path: Path) -> None:
    mask = np.zeros((9, 8, 7), dtype=np.uint8)
    mask[2:7, 2:6, 1:6] = 1
    path = tmp_path / "bone_mask.nii.gz"

    save_mask(
        path,
        mask,
        spacing_mm=(0.7, 0.7, 0.7),
        origin_lps_mm=(1.0, 2.0, 3.0),
    )
    loaded = load_volume(path, spacing_mm=0.7)

    np.testing.assert_allclose(loaded.data, mask.astype(np.float32), atol=1e-6)
    assert loaded.spacing_mm == pytest.approx((0.7, 0.7, 0.7))
    assert loaded.origin_lps_mm == pytest.approx((1.0, 2.0, 3.0))


def test_nifti_round_trip_does_not_add_voxels_at_point_eight_mm(
    tmp_path: Path,
) -> None:
    mask = np.zeros((17, 16, 15), dtype=np.uint8)
    mask[3:14, 3:13, 2:12] = 1
    path = tmp_path / "point_eight_mask.nii.gz"

    save_mask(path, mask, spacing_mm=(0.8, 0.8, 0.8))
    loaded = load_volume(path, spacing_mm=0.8, interpolation="nearest")

    assert loaded.data.shape == mask.shape
    np.testing.assert_array_equal(loaded.data, mask)


def test_unsupported_input_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scan.txt"
    path.write_text("not an image", encoding="ascii")
    with pytest.raises(ValueError, match="Unsupported"):
        load_volume(path)


def test_save_rejects_non_binary_mask(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        save_mask(
            tmp_path / "mask.nii.gz",
            np.full((2, 2, 2), 2, dtype=np.uint8),
            spacing_mm=(1.0, 1.0, 1.0),
        )


def test_ubd_reader_preserves_stored_spacing(tmp_path: Path) -> None:
    path = tmp_path / "scan.ubd.npz"
    body_data = BodyData()
    body_data.from_array(np.zeros((3, 3, 3), dtype=np.float32), "ct", 0.45)
    body_data.save(str(path))

    loaded = MedicalVolumeReader(spacing_mm=1.0).read(path)

    assert loaded.spacing_mm == pytest.approx((0.45, 0.45, 0.45))
