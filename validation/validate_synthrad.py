"""Run cbct-bone on paired SynthRAD2023 pelvis CBCT/CT cases."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from download_synthrad import CASES, OUTPUT_ROOT
from scipy import ndimage as ndi

from cbct_bone import (
    AdaptiveBoneSegmenter,
    BoneSegmentationConfig,
    MedicalVolumeReader,
)
from cbct_bone.io import MedicalVolume

RESULTS_ROOT = Path(__file__).parent / "results"
CT_PROXY_CONFIG = BoneSegmentationConfig(
    weak_threshold=150.0,
    strong_threshold=300.0,
)


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    case: str
    center: str
    dice_vs_ct_proxy: float
    sensitivity_vs_ct_proxy: float
    precision_vs_ct_proxy: float
    surface_dice_2mm_vs_ct_proxy: float
    hd95_mm_vs_ct_proxy: float
    cbct_bone_ml: float
    ct_bone_ml: float
    cbct_components: int
    ct_components: int
    weak_threshold_cbct: float
    strong_threshold_cbct: float
    seconds: float


def _dice(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    prediction = prediction.astype(bool, copy=False)
    reference = reference.astype(bool, copy=False)
    intersection = np.count_nonzero(prediction & reference)
    pred_count = np.count_nonzero(prediction)
    ref_count = np.count_nonzero(reference)
    dice = 2 * intersection / max(1, pred_count + ref_count)
    sensitivity = intersection / max(1, ref_count)
    precision = intersection / max(1, pred_count)
    return dice, sensitivity, precision


def _resample_to_reference(
    data: np.ndarray,
    source: MedicalVolume,
    reference: MedicalVolume,
    *,
    order: int,
) -> np.ndarray:
    """Resample an already-loaded canonical LPS array onto a reference grid."""

    source_spacing = np.asarray(source.spacing_mm, dtype=np.float64)
    reference_spacing = np.asarray(reference.spacing_mm, dtype=np.float64)
    source_origin = np.asarray(source.origin_lps_mm, dtype=np.float64)
    reference_origin = np.asarray(reference.origin_lps_mm, dtype=np.float64)
    matrix = np.diag(reference_spacing / source_spacing)
    offset = (reference_origin - source_origin) / source_spacing
    return ndi.affine_transform(
        data,
        matrix=matrix,
        offset=offset,
        output_shape=reference.data.shape,
        order=order,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def _surface_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    tolerance_mm: float = 2.0,
) -> tuple[float, float]:
    structure = ndi.generate_binary_structure(3, 1)
    union = prediction | reference
    if not np.any(union):
        return 0.0, float("inf")
    coordinates = np.where(union)
    slices = tuple(
        slice(max(0, int(axis.min()) - 1), min(union.shape[index], int(axis.max()) + 2))
        for index, axis in enumerate(coordinates)
    )
    prediction = prediction[slices]
    reference = reference[slices]
    pred_surface = prediction & ~ndi.binary_erosion(prediction, structure=structure)
    ref_surface = reference & ~ndi.binary_erosion(reference, structure=structure)
    if not np.any(pred_surface) or not np.any(ref_surface):
        return 0.0, float("inf")
    ref_distance = ndi.distance_transform_edt(~ref_surface, sampling=spacing_mm)
    pred_to_ref = np.asarray(ref_distance[pred_surface], dtype=np.float32)
    del ref_distance
    pred_distance = ndi.distance_transform_edt(~pred_surface, sampling=spacing_mm)
    ref_to_pred = pred_distance[ref_surface]
    surface_dice = float(
        (
            np.count_nonzero(pred_to_ref <= tolerance_mm)
            + np.count_nonzero(ref_to_pred <= tolerance_mm)
        )
        / (pred_to_ref.size + ref_to_pred.size)
    )
    hd95 = float(np.percentile(np.concatenate((pred_to_ref, ref_to_pred)), 95))
    return surface_dice, hd95


def _display_slice(volume: np.ndarray) -> np.ndarray:
    low, high = np.percentile(volume[np.isfinite(volume)], (1, 99))
    return np.clip((volume - low) / max(high - low, 1e-6), 0, 1)


def _representative_index(mask: np.ndarray, axis: int) -> int:
    counts = np.count_nonzero(mask, axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(counts))


def _make_case_figure(
    case: str,
    cbct: np.ndarray,
    prediction: np.ndarray,
    ct_reference: np.ndarray,
    metrics: CaseMetrics,
) -> None:
    axis = 2
    index = _representative_index(ct_reference, axis)
    image = _display_slice(cbct[:, :, index]).T
    pred = prediction[:, :, index].T.astype(bool)
    ref = ct_reference[:, :, index].T.astype(bool)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", origin="lower")
    axes[0].set_title(f"{case} CBCT")
    axes[1].imshow(image, cmap="gray", origin="lower")
    axes[1].contour(pred, levels=[0.5], colors=["#00d084"], linewidths=0.8)
    axes[1].set_title("Prediction (green)")
    axes[2].imshow(image, cmap="gray", origin="lower")
    axes[2].contour(ref, levels=[0.5], colors=["#ffd166"], linewidths=0.8)
    axes[2].contour(pred, levels=[0.5], colors=["#00d084"], linewidths=0.8)
    axes[2].set_title(f"CT proxy / prediction\nDice {metrics.dice_vs_ct_proxy:.3f}")
    for axis_item in axes:
        axis_item.axis("off")
    figure.savefig(RESULTS_ROOT / f"{case}_slices.png", dpi=160)
    plt.close(figure)


def _make_summary_figure(metrics: list[CaseMetrics]) -> None:
    labels = [item.case for item in metrics]
    dice = [item.dice_vs_ct_proxy for item in metrics]
    sensitivity = [item.sensitivity_vs_ct_proxy for item in metrics]
    precision = [item.precision_vs_ct_proxy for item in metrics]
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.bar(x - width, dice, width, label="Dice")
    axis.bar(x, sensitivity, width, label="Sensitivity")
    axis.bar(x + width, precision, width, label="Precision")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Overlap against paired CT proxy")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=3)
    figure.savefig(RESULTS_ROOT / "summary_metrics.png", dpi=160)
    plt.close(figure)


def _make_representative_figure(metrics: list[CaseMetrics]) -> None:
    representatives = [
        next(item for item in metrics if item.center == center)
        for center in ("A", "B", "C")
    ]
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for axis, item in zip(axes, representatives, strict=True):
        axis.imshow(plt.imread(RESULTS_ROOT / f"{item.case}_slices.png"))
        axis.set_title(f"Centre {item.center}: {item.case}")
        axis.axis("off")
    figure.savefig(RESULTS_ROOT / "representative_centres.png", dpi=140)
    plt.close(figure)


def _make_before_after_figure(case: str) -> None:
    baseline = RESULTS_ROOT / "baseline" / f"{case}_slices.png"
    improved = RESULTS_ROOT / f"{case}_slices.png"
    if not baseline.is_file():
        return
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].imshow(plt.imread(baseline))
    axes[0].set_title("Before: whole-volume histogram thresholds")
    axes[1].imshow(plt.imread(improved))
    axes[1].set_title("After: patient-only adaptive thresholds")
    for axis in axes:
        axis.axis("off")
    figure.savefig(RESULTS_ROOT / "threshold_fix_comparison.png", dpi=140)
    plt.close(figure)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    reader = MedicalVolumeReader(spacing_mm=1.0)
    segmenter = AdaptiveBoneSegmenter()
    ct_proxy_segmenter = AdaptiveBoneSegmenter(CT_PROXY_CONFIG)
    results: list[CaseMetrics] = []

    for case in CASES:
        started = time.perf_counter()
        case_dir = OUTPUT_ROOT / case
        cbct = reader.read(case_dir / "cbct.nii.gz")
        ct = reader.read(case_dir / "ct.nii.gz")
        patient_outline = reader.read(case_dir / "mask.nii.gz")
        prediction, cbct_diagnostics = segmenter.segment(cbct.data, cbct.spacing_mm)
        ct_reference, ct_diagnostics = ct_proxy_segmenter.segment(
            ct.data, ct.spacing_mm
        )
        ct_reference = _resample_to_reference(ct_reference, ct, cbct, order=0).astype(
            bool
        )
        support = (
            _resample_to_reference(patient_outline.data, patient_outline, cbct, order=0)
            > 0.5
        )
        prediction = prediction.astype(bool) & support
        ct_reference &= support
        dice, sensitivity, precision = _dice(prediction, ct_reference)
        surface_dice, hd95 = _surface_metrics(prediction, ct_reference, cbct.spacing_mm)
        voxel_ml = float(np.prod(cbct.spacing_mm)) / 1000.0
        metrics = CaseMetrics(
            case=case,
            center=case[2],
            dice_vs_ct_proxy=dice,
            sensitivity_vs_ct_proxy=sensitivity,
            precision_vs_ct_proxy=precision,
            surface_dice_2mm_vs_ct_proxy=surface_dice,
            hd95_mm_vs_ct_proxy=hd95,
            cbct_bone_ml=float(np.count_nonzero(prediction) * voxel_ml),
            ct_bone_ml=float(np.count_nonzero(ct_reference) * voxel_ml),
            cbct_components=cbct_diagnostics.component_count,
            ct_components=ct_diagnostics.component_count,
            weak_threshold_cbct=cbct_diagnostics.weak_threshold,
            strong_threshold_cbct=cbct_diagnostics.strong_threshold,
            seconds=time.perf_counter() - started,
        )
        results.append(metrics)
        _make_case_figure(
            case,
            cbct.data,
            prediction,
            ct_reference,
            metrics,
        )
        print(json.dumps(asdict(metrics), sort_keys=True))

    metrics_csv = RESULTS_ROOT / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)
    (RESULTS_ROOT / "metrics.json").write_text(
        json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8"
    )
    _make_summary_figure(results)
    _make_representative_figure(results)
    _make_before_after_figure("2PA024")


if __name__ == "__main__":
    main()
