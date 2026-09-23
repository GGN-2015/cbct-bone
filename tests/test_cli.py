from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from cbct_bone.algorithm import SegmentationDiagnostics
from cbct_bone.api import SegmentationResult
from cbct_bone.cli import main
from cbct_bone.io import MedicalVolume


def _result(tmp_path: Path) -> SegmentationResult:
    volume = MedicalVolume(
        data=np.zeros((2, 2, 2), dtype=np.float32),
        spacing_mm=(0.8, 0.8, 0.8),
        origin_lps_mm=(0.0, 0.0, 0.0),
        metadata={},
        source=tmp_path / "input",
    )
    return SegmentationResult(
        mask=np.ones((2, 2, 2), dtype=np.uint8),
        volume=volume,
        diagnostics=SegmentationDiagnostics(100.0, 300.0, 8, 1),
        output_path=(tmp_path / "mask.nii.gz").resolve(),
    )


def test_cli_prints_json_diagnostics(tmp_path: Path, capsys) -> None:
    with patch("cbct_bone.cli.segment", return_value=_result(tmp_path)) as mocked:
        status = main(["input", str(tmp_path / "mask.nii.gz"), "--json"])

    assert status == 0
    assert '"bone_voxels": 8' in capsys.readouterr().out
    assert mocked.call_args.kwargs["spacing_mm"] == 0.8
    assert mocked.call_args.kwargs["backend"] == "v4"
    assert mocked.call_args.kwargs["neural_config"].batch_size == 8


def test_cli_selects_adaptive_backend(tmp_path: Path) -> None:
    with patch("cbct_bone.cli.segment", return_value=_result(tmp_path)) as mocked:
        status = main(
            [
                "input",
                str(tmp_path / "mask.nii.gz"),
                "--backend",
                "adaptive",
                "--weak-threshold",
                "100",
                "--strong-threshold",
                "300",
            ]
        )

    assert status == 0
    assert mocked.call_args.kwargs["backend"] == "adaptive"
    assert mocked.call_args.kwargs["config"].weak_threshold == 100.0
    assert mocked.call_args.kwargs["neural_config"] is None


def test_cli_passes_explicit_annotation_json(tmp_path: Path) -> None:
    annotation_path = tmp_path / "editable.json"
    with patch("cbct_bone.cli.segment", return_value=_result(tmp_path)) as mocked:
        status = main(
            [
                "input",
                str(tmp_path / "mask.nii.gz"),
                "--annotation-json",
                str(annotation_path),
            ]
        )

    assert status == 0
    assert mocked.call_args.kwargs["annotation_path"] == annotation_path


def test_cli_annotation_json_without_path_uses_colocated_name(tmp_path: Path) -> None:
    source = tmp_path / "scan.nii.gz"
    source.write_bytes(b"source")
    with patch("cbct_bone.cli.segment", return_value=_result(tmp_path)) as mocked:
        status = main(
            [
                str(source),
                str(tmp_path / "mask.nii.gz"),
                "--annotation-json",
            ]
        )

    assert status == 0
    assert mocked.call_args.kwargs["annotation_path"] == (
        tmp_path / "scan.med-image-seg.json"
    )


def test_cli_reports_expected_errors(tmp_path: Path, capsys) -> None:
    with patch("cbct_bone.cli.segment", side_effect=ValueError("bad volume")):
        status = main(["input", str(tmp_path / "mask.nii.gz")])

    assert status == 2
    assert "bad volume" in capsys.readouterr().err
