"""Command-line interface for cbct-bone."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .algorithm import BoneSegmentationConfig
from .annotations import annotation_path_for
from .api import segment
from .inference import NeuralSegmentationConfig, NeuralSegmentationDiagnostics
from .models import DownloadProgress

_AUTO_ANNOTATION_PATH = Path("__AUTO_MED_IMAGE_SEG_SIDECAR__")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbct-bone",
        description="Extract a binary bone mask from a CT/CBCT volume.",
    )
    parser.add_argument(
        "input", type=Path, help="DICOM directory/file or NIfTI/UBD file"
    )
    parser.add_argument("output", type=Path, help="Output .nii.gz, .nii, or .ubd.npz")
    parser.add_argument(
        "--annotation-json",
        nargs="?",
        type=Path,
        const=_AUTO_ANNOTATION_PATH,
        metavar="PATH",
        help=(
            "also write editable polygon JSON; omit PATH to place it beside the input"
        ),
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.8,
        metavar="MM",
        help="isotropic processing spacing in mm; v4 requires 0.8 (default: 0.8)",
    )
    parser.add_argument(
        "--backend",
        choices=("v4", "adaptive"),
        default="v4",
        help="segmentation backend (default: v4)",
    )
    parser.add_argument("--model", type=Path, help="use a local v4 ONNX model")
    parser.add_argument("--cache-dir", type=Path, help="model cache directory")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not download a missing v4 model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="v4 inference batch size (default: 8)",
    )
    parser.add_argument(
        "--provider",
        action="append",
        help="ONNX Runtime provider; repeat to define provider priority",
    )
    adaptive = parser.add_argument_group("adaptive backend options")
    adaptive.add_argument("--weak-threshold", type=float, default=None)
    adaptive.add_argument("--strong-threshold", type=float, default=None)
    adaptive.add_argument("--local-radius", type=float, default=2.0, metavar="MM")
    adaptive.add_argument("--local-sigma", type=float, default=0.25)
    adaptive.add_argument("--closing-radius", type=float, default=1.0, metavar="MM")
    adaptive.add_argument("--minimum-component", type=float, default=8.0, metavar="MM3")
    adaptive.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="keep marrow cavities instead of filling enclosed bone surfaces",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable diagnostics",
    )
    return parser


def _download_progress(progress: DownloadProgress) -> None:
    total = max(1, progress.total_bytes)
    percent = min(100.0, 100.0 * progress.downloaded_bytes / total)
    end = "\n" if progress.downloaded_bytes >= progress.total_bytes else ""
    print(
        f"\rDownloading v4 model: {percent:5.1f}%",
        end=end,
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    annotation_path = args.annotation_json
    if annotation_path == _AUTO_ANNOTATION_PATH:
        try:
            annotation_path = annotation_path_for(args.input)
        except (OSError, ValueError) as exc:
            print(f"cbct-bone: error: {exc}", file=sys.stderr)
            return 2
    config = (
        BoneSegmentationConfig(
            weak_threshold=args.weak_threshold,
            strong_threshold=args.strong_threshold,
            local_radius_mm=args.local_radius,
            local_sigma=args.local_sigma,
            closing_radius_mm=args.closing_radius,
            minimum_component_mm3=args.minimum_component,
            fill_holes=not args.no_fill_holes,
        )
        if args.backend == "adaptive"
        else None
    )
    neural_config = (
        NeuralSegmentationConfig(batch_size=args.batch_size)
        if args.backend == "v4"
        else None
    )
    try:
        result = segment(
            args.input,
            args.output,
            spacing_mm=args.spacing,
            backend=args.backend,
            config=config,
            neural_config=neural_config,
            model_path=args.model,
            cache_dir=args.cache_dir,
            offline=args.offline,
            providers=args.provider,
            download_progress=_download_progress,
            annotation_path=annotation_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"cbct-bone: error: {exc}", file=sys.stderr)
        return 2

    diagnostics = asdict(result.diagnostics)
    diagnostics["output"] = str(result.output_path)
    diagnostics["spacing_mm"] = result.volume.spacing_mm
    diagnostics["annotation_json"] = (
        str(result.annotation_path) if result.annotation_path is not None else None
    )
    if args.json:
        print(json.dumps(diagnostics, ensure_ascii=True, sort_keys=True))
    elif isinstance(result.diagnostics, NeuralSegmentationDiagnostics):
        print(
            f"Saved {diagnostics['bone_voxels']} bone voxels to "
            f"{diagnostics['output']} using "
            f"{diagnostics['model_name']} {diagnostics['model_version']} "
            f"({diagnostics['provider']})."
        )
    else:
        print(
            f"Saved {diagnostics['bone_voxels']} bone voxels to "
            f"{diagnostics['output']} "
            f"(thresholds {diagnostics['weak_threshold']:.3g}/"
            f"{diagnostics['strong_threshold']:.3g})."
        )
    if result.annotation_path is not None and not args.json:
        print(f"Saved editable polygons to {result.annotation_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
