"""Device-specific CBCT bone mask extraction."""

from .algorithm import (
    AdaptiveBoneSegmenter,
    BoneSegmentationConfig,
    SegmentationDiagnostics,
    segment_array,
)
from .annotations import (
    MachineAnnotationWriter,
    annotation_path_for,
    source_fingerprint,
)
from .api import (
    AnnotationWriter,
    CBCTBoneExtractor,
    Diagnostics,
    SegmentationBackend,
    SegmentationResult,
    segment,
)
from .inference import (
    NeuralCBCTBoneSegmenter,
    NeuralMaskPostprocessor,
    NeuralSegmentationConfig,
    NeuralSegmentationDiagnostics,
    V4CBCTBoneSegmenter,
    V5CBCTBoneSegmenter,
)
from .io import (
    MedicalMaskWriter,
    MedicalVolume,
    MedicalVolumeReader,
    load_volume,
    save_mask,
)
from .models import (
    LATEST_MODEL,
    V4_MODEL,
    V5_MODEL,
    DownloadProgress,
    ModelError,
    ModelIntegrityError,
    ModelManager,
    ModelSpec,
    ModelUnavailableError,
    default_model_cache,
)

__all__ = [
    "LATEST_MODEL",
    "V4_MODEL",
    "V5_MODEL",
    "AdaptiveBoneSegmenter",
    "AnnotationWriter",
    "BoneSegmentationConfig",
    "CBCTBoneExtractor",
    "Diagnostics",
    "DownloadProgress",
    "MachineAnnotationWriter",
    "MedicalMaskWriter",
    "MedicalVolume",
    "MedicalVolumeReader",
    "ModelError",
    "ModelIntegrityError",
    "ModelManager",
    "ModelSpec",
    "ModelUnavailableError",
    "NeuralCBCTBoneSegmenter",
    "NeuralMaskPostprocessor",
    "NeuralSegmentationConfig",
    "NeuralSegmentationDiagnostics",
    "SegmentationBackend",
    "SegmentationDiagnostics",
    "SegmentationResult",
    "V4CBCTBoneSegmenter",
    "V5CBCTBoneSegmenter",
    "annotation_path_for",
    "default_model_cache",
    "load_volume",
    "save_mask",
    "segment",
    "segment_array",
    "source_fingerprint",
]

__version__ = "0.3.0"
