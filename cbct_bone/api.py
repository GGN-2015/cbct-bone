"""High-level public API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .algorithm import (
    AdaptiveBoneSegmenter,
    BoneSegmentationConfig,
    SegmentationDiagnostics,
)
from .annotations import MachineAnnotationWriter
from .inference import (
    NeuralSegmentationConfig,
    NeuralSegmentationDiagnostics,
    V4CBCTBoneSegmenter,
    V5CBCTBoneSegmenter,
)
from .io import MedicalMaskWriter, MedicalVolume, MedicalVolumeReader
from .models import V4_MODEL, V5_MODEL, ModelManager, ProgressCallback

SegmentationBackend = Literal["v5", "v4", "adaptive"]
Diagnostics = SegmentationDiagnostics | NeuralSegmentationDiagnostics


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Bone mask and the geometry/diagnostics needed to use it safely."""

    mask: np.ndarray
    volume: MedicalVolume
    diagnostics: Diagnostics
    output_path: Path | None = None
    annotation_path: Path | None = None


class VolumeReader(Protocol):
    """Interface for injectable volume readers."""

    def read(
        self,
        path: str | Path,
        *,
        spacing_mm: float | None = None,
        interpolation: Literal["linear", "nearest"] = "linear",
    ) -> MedicalVolume: ...


class BoneMaskSegmenter(Protocol):
    """Interface for interchangeable bone segmentation algorithms."""

    def segment(
        self,
        volume: np.ndarray,
        spacing_mm: float | tuple[float, float, float] = 1.0,
    ) -> tuple[np.ndarray, Diagnostics]: ...


class MaskWriter(Protocol):
    """Interface for injectable mask writers."""

    def write(
        self,
        path: str | Path,
        mask: np.ndarray,
        reference: MedicalVolume,
    ) -> Path: ...


class AnnotationWriter(Protocol):
    """Interface for editable polygon sidecar writers."""

    def write(
        self,
        path: str | Path,
        mask: np.ndarray,
        reference: MedicalVolume,
        *,
        source_path: str | Path | None = None,
    ) -> Path: ...


class CBCTBoneExtractor:
    """Coordinate medical image reading, segmentation, and mask writing.

    Dependencies are injected through small interfaces, so downstream projects
    can replace any stage without subclassing this coordinator.
    """

    def __init__(
        self,
        *,
        reader: VolumeReader | None = None,
        segmenter: BoneMaskSegmenter | None = None,
        writer: MaskWriter | None = None,
        annotation_writer: AnnotationWriter | None = None,
    ) -> None:
        self.reader = reader if reader is not None else MedicalVolumeReader()
        self.segmenter = segmenter if segmenter is not None else V5CBCTBoneSegmenter()
        self.writer = writer if writer is not None else MedicalMaskWriter()
        self.annotation_writer = (
            annotation_writer
            if annotation_writer is not None
            else MachineAnnotationWriter()
        )

    def extract(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        spacing_mm: float | None = None,
        annotation_path: str | Path | None = None,
    ) -> SegmentationResult:
        """Extract bone from one medical image and optionally write the mask."""

        volume = self.reader.read(input_path, spacing_mm=spacing_mm)
        mask, diagnostics = self.segmenter.segment(
            volume.data,
            spacing_mm=volume.spacing_mm,
        )
        saved_path = (
            self.writer.write(output_path, mask, volume)
            if output_path is not None
            else None
        )
        saved_annotation_path = (
            self.annotation_writer.write(
                annotation_path,
                mask,
                volume,
                source_path=input_path,
            )
            if annotation_path is not None
            else None
        )
        return SegmentationResult(
            mask=mask,
            volume=volume,
            diagnostics=diagnostics,
            output_path=saved_path,
            annotation_path=saved_annotation_path,
        )


def segment(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    spacing_mm: float = 0.8,
    backend: SegmentationBackend = "v5",
    config: BoneSegmentationConfig | None = None,
    neural_config: NeuralSegmentationConfig | None = None,
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    providers: Sequence[str] | None = None,
    download_progress: ProgressCallback | None = None,
    annotation_path: str | Path | None = None,
) -> SegmentationResult:
    """Convenience wrapper around :class:`CBCTBoneExtractor`."""

    if backend not in ("v5", "v4", "adaptive"):
        raise ValueError("backend must be 'v5', 'v4', or 'adaptive'")
    if backend in ("v5", "v4"):
        if config is not None:
            raise ValueError("config applies only to the adaptive backend")
        model_spec = V5_MODEL if backend == "v5" else V4_MODEL
        segmenter_type = V5CBCTBoneSegmenter if backend == "v5" else V4CBCTBoneSegmenter
        manager = (
            ModelManager(model_spec, cache_dir=cache_dir)
            if model_path is None
            else None
        )
        segmenter: BoneMaskSegmenter = segmenter_type(
            neural_config,
            model_manager=manager,
            model_path=model_path,
            offline=offline,
            providers=providers,
            download_progress=download_progress,
        )
    else:
        if (
            neural_config is not None
            or model_path is not None
            or cache_dir is not None
            or offline
            or providers
        ):
            raise ValueError("neural model options require backend='v5' or 'v4'")
        segmenter = AdaptiveBoneSegmenter(config)
    extractor = CBCTBoneExtractor(
        reader=MedicalVolumeReader(spacing_mm),
        segmenter=segmenter,
    )
    return extractor.extract(
        input_path,
        output_path,
        annotation_path=annotation_path,
    )


__all__ = [
    "AnnotationWriter",
    "BoneMaskSegmenter",
    "CBCTBoneExtractor",
    "Diagnostics",
    "MaskWriter",
    "SegmentationBackend",
    "SegmentationResult",
    "VolumeReader",
    "segment",
]
