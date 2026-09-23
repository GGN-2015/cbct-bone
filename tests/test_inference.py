from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cbct_bone.inference import (
    NeuralSegmentationConfig,
    V4CBCTBoneSegmenter,
)


@dataclass
class TensorInfo:
    name: str


class FakeSession:
    def get_inputs(self) -> list[TensorInfo]:
        return [TensorInfo("images")]

    def get_outputs(self) -> list[TensorInfo]:
        return [TensorInfo("probabilities")]

    def get_providers(self) -> list[str]:
        return ["FakeExecutionProvider"]

    def run(
        self,
        _output_names: list[str],
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        inputs = input_feed["images"]
        return [inputs[:, 1:2].copy()]


def _segmenter() -> V4CBCTBoneSegmenter:
    config = NeuralSegmentationConfig(
        batch_size=2,
        threshold=0.5,
        minimum_component_pixels=4,
        maximum_hole_pixels=0,
        closing_radius_pixels=0,
    )
    return V4CBCTBoneSegmenter(config, session=FakeSession())


def test_v4_segmenter_restores_lps_axis_order() -> None:
    volume = np.zeros((16, 12, 3), dtype=np.float32)
    volume[2:12, 3:10, :] = 100.0

    mask, diagnostics = _segmenter().segment(volume, (0.8, 0.8, 0.8))

    expected = np.zeros_like(volume, dtype=np.uint8)
    expected[2:12, 3:10, :] = 1
    assert np.array_equal(mask, expected)
    assert diagnostics.bone_voxels == int(expected.sum())
    assert diagnostics.slice_count == 3
    assert diagnostics.provider == "FakeExecutionProvider"


def test_v4_segmenter_rejects_wrong_spacing() -> None:
    volume = np.zeros((8, 8, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"expects 0\.8 mm"):
        _segmenter().segment(volume, (1.0, 1.0, 1.0))


def test_v4_config_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        NeuralSegmentationConfig(batch_size=0)
