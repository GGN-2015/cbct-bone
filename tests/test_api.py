from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np

from cbct_bone.algorithm import SegmentationDiagnostics
from cbct_bone.api import CBCTBoneExtractor
from cbct_bone.inference import V4CBCTBoneSegmenter
from cbct_bone.io import MedicalVolume


def test_segment_high_level_api(tmp_path: Path) -> None:
    data = np.full((24, 24, 24), -900.0, dtype=np.float32)
    data[5:19, 5:19, 5:19] = 50.0
    data[8:16, 8:16, 8:16] = 900.0
    volume = MedicalVolume(
        data=data,
        spacing_mm=(1.0, 1.0, 1.0),
        origin_lps_mm=(0.0, 0.0, 0.0),
        metadata={"modality": "CBCT"},
        source=tmp_path / "input",
    )

    reader = Mock()
    reader.read.return_value = volume
    segmenter = Mock()
    segmenter.segment.return_value = (
        np.ones_like(data, dtype=np.uint8),
        SegmentationDiagnostics(100.0, 500.0, data.size, 1),
    )
    writer = Mock()
    writer.write.return_value = (tmp_path / "mask.nii.gz").resolve()
    annotation_writer = Mock()
    annotation_writer.write.return_value = (tmp_path / "scan.json").resolve()

    extractor = CBCTBoneExtractor(
        reader=reader,
        segmenter=segmenter,
        writer=writer,
        annotation_writer=annotation_writer,
    )
    result = extractor.extract(
        "unused",
        tmp_path / "mask.nii.gz",
        annotation_path=tmp_path / "scan.json",
    )

    assert result.output_path == (tmp_path / "mask.nii.gz").resolve()
    assert result.mask.dtype == np.uint8
    assert result.mask.sum() > 0
    assert result.annotation_path == (tmp_path / "scan.json").resolve()
    reader.read.assert_called_once_with("unused", spacing_mm=None)
    segmenter.segment.assert_called_once()
    writer.write.assert_called_once_with(tmp_path / "mask.nii.gz", result.mask, volume)
    annotation_writer.write.assert_called_once_with(
        tmp_path / "scan.json",
        result.mask,
        volume,
        source_path="unused",
    )


def test_extractor_defaults_to_v4_without_eager_download() -> None:
    extractor = CBCTBoneExtractor()

    assert isinstance(extractor.segmenter, V4CBCTBoneSegmenter)
    assert extractor.segmenter._model_path is None
