"""An anatomy-independent 3-D adaptive bone segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.morphology import ball


@dataclass(frozen=True, slots=True)
class BoneSegmentationConfig:
    """Parameters for adaptive bone segmentation.

    Physical-size options are expressed in millimetres so that results do not
    depend on the voxel resolution selected while loading the image.
    """

    weak_threshold: float | None = None
    strong_threshold: float | None = None
    local_radius_mm: float = 2.0
    local_sigma: float = 0.25
    closing_radius_mm: float = 1.0
    minimum_component_mm3: float = 8.0
    fill_holes: bool = True
    connectivity: Literal[1, 2, 3] = 2
    max_histogram_samples: int = 1_000_000

    def __post_init__(self) -> None:
        if self.weak_threshold is not None and not np.isfinite(self.weak_threshold):
            raise ValueError("weak_threshold must be finite")
        if self.strong_threshold is not None and not np.isfinite(self.strong_threshold):
            raise ValueError("strong_threshold must be finite")
        if (
            self.weak_threshold is not None
            and self.strong_threshold is not None
            and self.weak_threshold >= self.strong_threshold
        ):
            raise ValueError("weak_threshold must be lower than strong_threshold")
        if self.local_radius_mm <= 0:
            raise ValueError("local_radius_mm must be greater than zero")
        if self.local_sigma < 0:
            raise ValueError("local_sigma must not be negative")
        if self.closing_radius_mm < 0:
            raise ValueError("closing_radius_mm must not be negative")
        if self.minimum_component_mm3 < 0:
            raise ValueError("minimum_component_mm3 must not be negative")
        if self.connectivity not in (1, 2, 3):
            raise ValueError("connectivity must be 1, 2, or 3")
        if self.max_histogram_samples < 1_000:
            raise ValueError("max_histogram_samples must be at least 1000")


@dataclass(frozen=True, slots=True)
class SegmentationDiagnostics:
    """Thresholds and counts recorded for one segmentation."""

    weak_threshold: float
    strong_threshold: float
    bone_voxels: int
    component_count: int


def _as_spacing(spacing_mm: float | tuple[float, float, float]) -> np.ndarray:
    spacing = np.asarray(
        (spacing_mm, spacing_mm, spacing_mm) if np.isscalar(spacing_mm) else spacing_mm,
        dtype=np.float64,
    )
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)):
        raise ValueError("spacing_mm must contain three finite values")
    if np.any(spacing <= 0):
        raise ValueError("spacing_mm values must be greater than zero")
    return spacing


def _sample_finite(
    volume: np.ndarray, max_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(volume)
    values = volume[finite]
    if values.size < 3:
        raise ValueError("volume must contain at least three finite voxels")
    stride = max(1, values.size // max_samples)
    sample = np.asarray(values[::stride], dtype=np.float32)
    low, high = np.percentile(sample, (0.1, 99.9))
    sample = sample[(sample >= low) & (sample <= high)]
    if np.unique(sample).size < 3:
        raise ValueError("volume does not have enough intensity variation")
    return finite, sample


def estimate_thresholds(
    volume: np.ndarray,
    *,
    max_samples: int = 1_000_000,
) -> tuple[float, float]:
    """Estimate bone thresholds after excluding the scanner-air intensity mode.

    CBCT grey values are not calibrated across scanners. The estimator first
    separates air/background from the patient, then expresses the bone cutoffs
    as percentiles of the patient-only histogram. A five-class Otsu partition
    identifies whether the brightest class is a genuine cortical-bone tail or
    a tiny metal/artifact tail.
    """

    _, sample = _sample_finite(volume, max_samples)
    try:
        air_t = float(threshold_multiotsu(sample, classes=3, nbins=256)[0])
    except ValueError:
        (air_t,) = threshold_multiotsu(sample, classes=2, nbins=256)

    patient = sample[sample > air_t]
    if patient.size < 1_000 or np.unique(patient).size < 5:
        patient = sample

    try:
        partitions = threshold_multiotsu(patient, classes=5, nbins=256)
        upper_boundary = float(partitions[-1])
        percentile_995 = float(np.percentile(patient, 99.5))
        if upper_boundary > percentile_995:
            # A tiny metal or clipping mode should not define the bone class.
            upper_boundary = float(partitions[-2])
        boundary_rank = 100.0 * float(np.mean(patient <= upper_boundary))
        weak_percentile = float(np.clip(2.0 * boundary_rank - 100.0, 90.0, 98.0))
        strong_percentile = float(np.clip(max(98.5, weak_percentile + 1.5), 98.5, 99.5))
        weak_t, strong_t = np.percentile(patient, (weak_percentile, strong_percentile))
    except ValueError:
        weak_t, strong_t = np.percentile(patient, (92.0, 99.0))

    weak_t = float(weak_t)
    strong_t = float(strong_t)
    if weak_t >= strong_t:
        weak_t = float(np.nextafter(np.float32(strong_t), np.float32(-np.inf)))
    return weak_t, strong_t


def _local_candidate_mask(
    volume: np.ndarray,
    finite: np.ndarray,
    weak_threshold: float,
    strong_threshold: float,
    radius_voxels: np.ndarray,
    local_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    size = tuple(int(2 * radius + 1) for radius in radius_voxels)
    working = np.where(finite, volume, weak_threshold).astype(np.float32, copy=False)
    local_mean = ndi.uniform_filter(working, size=size, mode="nearest")
    local_square_mean = ndi.uniform_filter(
        np.square(working, dtype=np.float32), size=size, mode="nearest"
    )
    local_variance = np.maximum(local_square_mean - np.square(local_mean), 0.0)
    local_threshold = local_mean + local_sigma * np.sqrt(local_variance)

    strong = finite & (volume >= strong_threshold)
    candidate = (
        finite & (volume >= weak_threshold) & ((volume >= local_threshold) | strong)
    )
    return strong, candidate


def _remove_small_components(
    mask: np.ndarray,
    minimum_voxels: int,
    structure: np.ndarray,
) -> tuple[np.ndarray, int]:
    labels, count = ndi.label(mask, structure=structure)
    if count == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_voxels
    keep[0] = False
    cleaned = keep[labels]
    return cleaned, int(np.count_nonzero(keep))


class AdaptiveBoneSegmenter:
    """Reusable whole-body bone segmenter based on 3-D adaptive thresholding."""

    def __init__(self, config: BoneSegmentationConfig | None = None) -> None:
        self._config = config or BoneSegmentationConfig()

    @property
    def config(self) -> BoneSegmentationConfig:
        return self._config

    def estimate_thresholds(self, volume: np.ndarray) -> tuple[float, float]:
        """Estimate this segmenter's weak and strong thresholds."""

        return estimate_thresholds(
            volume,
            max_samples=self._config.max_histogram_samples,
        )

    def segment(
        self,
        volume: np.ndarray,
        spacing_mm: float | tuple[float, float, float] = 1.0,
    ) -> tuple[np.ndarray, SegmentationDiagnostics]:
        """Extract a binary bone mask from a 3-D CT/CBCT NumPy volume."""

        config = self._config
        array = np.asarray(volume)
        if array.ndim != 3:
            raise ValueError(f"volume must be 3-D, got shape {array.shape}")
        if array.size == 0:
            raise ValueError("volume must not be empty")
        spacing = _as_spacing(spacing_mm)
        finite = np.isfinite(array)
        if np.count_nonzero(finite) < 3:
            raise ValueError("volume must contain at least three finite voxels")

        estimated_weak: float | None = None
        estimated_strong: float | None = None
        if config.weak_threshold is None or config.strong_threshold is None:
            estimated_weak, estimated_strong = self.estimate_thresholds(array)
        weak = (
            float(config.weak_threshold)
            if config.weak_threshold is not None
            else estimated_weak
        )
        strong = (
            float(config.strong_threshold)
            if config.strong_threshold is not None
            else estimated_strong
        )
        assert weak is not None and strong is not None
        if weak >= strong:
            raise ValueError(
                f"weak threshold ({weak:g}) must be lower than strong threshold "
                f"({strong:g})"
            )

        radius_voxels = np.maximum(1, np.rint(config.local_radius_mm / spacing)).astype(
            int
        )
        seeds, candidates = _local_candidate_mask(
            array,
            finite,
            weak,
            strong,
            radius_voxels,
            config.local_sigma,
        )
        structure = ndi.generate_binary_structure(3, config.connectivity)
        mask = ndi.binary_propagation(seeds, structure=structure, mask=candidates)

        if config.closing_radius_mm > 0:
            radius = max(1, round(config.closing_radius_mm / float(spacing.min())))
            mask = ndi.binary_closing(mask, structure=ball(radius))
        if config.fill_holes:
            mask = ndi.binary_fill_holes(mask)

        voxel_volume = float(np.prod(spacing))
        minimum_voxels = max(
            1, int(np.ceil(config.minimum_component_mm3 / voxel_volume))
        )
        mask, component_count = _remove_small_components(
            mask, minimum_voxels, structure
        )
        output = np.asarray(mask, dtype=np.uint8)
        diagnostics = SegmentationDiagnostics(
            weak_threshold=weak,
            strong_threshold=strong,
            bone_voxels=int(output.sum()),
            component_count=component_count,
        )
        return output, diagnostics


def segment_array(
    volume: np.ndarray,
    spacing_mm: float | tuple[float, float, float] = 1.0,
    config: BoneSegmentationConfig | None = None,
) -> tuple[np.ndarray, SegmentationDiagnostics]:
    """Convenience wrapper around :class:`AdaptiveBoneSegmenter`."""

    return AdaptiveBoneSegmenter(config).segment(volume, spacing_mm)


__all__ = [
    "AdaptiveBoneSegmenter",
    "BoneSegmentationConfig",
    "SegmentationDiagnostics",
    "estimate_thresholds",
    "segment_array",
]
