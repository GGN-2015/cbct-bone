from __future__ import annotations

import numpy as np
import pytest

from cbct_bone.algorithm import BoneSegmentationConfig, segment_array


def _synthetic_joint(
    shape: tuple[int, int, int] = (80, 80, 64),
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.indices(shape, dtype=np.float32)
    x, y, z = grid
    expected = np.zeros(shape, dtype=bool)
    marrow = np.zeros(shape, dtype=bool)

    for center_z in (18.0, 46.0):
        outer = ((x - 40.0) / 18.0) ** 2 + ((y - 40.0) / 15.0) ** 2 + (
            (z - center_z) / 12.0
        ) ** 2 <= 1.0
        inner = ((x - 40.0) / 12.0) ** 2 + ((y - 40.0) / 9.0) ** 2 + (
            (z - center_z) / 8.0
        ) ** 2 <= 1.0
        expected |= outer
        marrow |= inner

    volume = np.full(shape, -900.0, dtype=np.float32)
    body = ((x - 40.0) / 34.0) ** 2 + ((y - 40.0) / 31.0) ** 2 <= 1.0
    volume[body] = 40.0
    volume[expected] = 750.0
    # Lower-density marrow is enclosed by a high-density cortical shell.
    volume[marrow] = 120.0
    rng = np.random.default_rng(42)
    volume += rng.normal(0.0, 22.0, size=shape).astype(np.float32)
    return volume, expected


def test_segment_array_extracts_synthetic_joint() -> None:
    volume, expected = _synthetic_joint()
    mask, diagnostics = segment_array(
        volume,
        spacing_mm=0.8,
        config=BoneSegmentationConfig(minimum_component_mm3=20.0),
    )

    intersection = np.count_nonzero(mask & expected)
    dice = 2 * intersection / (np.count_nonzero(mask) + np.count_nonzero(expected))
    assert mask.dtype == np.uint8
    assert mask.shape == volume.shape
    assert dice > 0.90
    assert diagnostics.bone_voxels == int(mask.sum())
    assert diagnostics.component_count == 2
    assert diagnostics.weak_threshold < diagnostics.strong_threshold


def test_segment_array_handles_positive_cbct_grey_value_shift() -> None:
    volume, expected = _synthetic_joint()
    shifted = (volume + 900.0) * 0.8

    mask, diagnostics = segment_array(shifted, spacing_mm=0.8)

    intersection = np.count_nonzero(mask & expected)
    dice = 2 * intersection / (np.count_nonzero(mask) + np.count_nonzero(expected))
    assert dice > 0.90
    assert diagnostics.weak_threshold > 100.0


def test_manual_thresholds_are_used() -> None:
    volume = np.zeros((24, 24, 24), dtype=np.float32)
    volume[6:18, 6:18, 6:18] = 900
    config = BoneSegmentationConfig(weak_threshold=250, strong_threshold=500)
    _, diagnostics = segment_array(volume, 1.0, config)
    assert diagnostics.weak_threshold == 250
    assert diagnostics.strong_threshold == 500


def test_segment_array_extracts_pelvis_like_ring() -> None:
    shape = (96, 72, 64)
    x, y, z = np.indices(shape, dtype=np.float32)
    left_outer = ((x - 30) / 20) ** 2 + ((y - 36) / 25) ** 2 + ((z - 32) / 22) ** 2 <= 1
    right_outer = ((x - 66) / 20) ** 2 + ((y - 36) / 25) ** 2 + (
        (z - 32) / 22
    ) ** 2 <= 1
    left_inner = ((x - 30) / 13) ** 2 + ((y - 36) / 18) ** 2 + ((z - 32) / 15) ** 2 <= 1
    right_inner = ((x - 66) / 13) ** 2 + ((y - 36) / 18) ** 2 + (
        (z - 32) / 15
    ) ** 2 <= 1
    expected = left_outer | right_outer
    volume = np.full(shape, -950, dtype=np.float32)
    body = ((x - 48) / 45) ** 2 + ((y - 36) / 33) ** 2 <= 1
    volume[body] = 35
    volume[expected] = 820
    volume[left_inner | right_inner] = 100

    mask, _ = segment_array(volume, spacing_mm=1.0)
    intersection = np.count_nonzero(mask & expected)
    dice = 2 * intersection / (np.count_nonzero(mask) + np.count_nonzero(expected))
    assert dice > 0.92


@pytest.mark.parametrize(
    ("volume", "message"),
    [
        (np.zeros((3, 3)), "3-D"),
        (np.zeros((3, 3, 3)), "variation"),
    ],
)
def test_invalid_volume_is_rejected(volume: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        segment_array(volume)


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="lower"):
        BoneSegmentationConfig(weak_threshold=500, strong_threshold=200)
