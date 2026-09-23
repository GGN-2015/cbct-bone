"""ONNX inference for the device-specific v4 CBCT bone model."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.ndimage import binary_closing, label
from skimage.morphology import disk

from .models import V4_MODEL, ModelManager, ProgressCallback


class InferenceSession(Protocol):
    """Subset of the ONNX Runtime session API used by this package."""

    def get_inputs(self) -> Sequence[Any]: ...

    def get_outputs(self) -> Sequence[Any]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(
        self,
        output_names: Sequence[str],
        input_feed: dict[str, np.ndarray],
    ) -> Sequence[np.ndarray]: ...


SessionFactory = Callable[[Path, tuple[str, ...] | None], InferenceSession]


@dataclass(frozen=True, slots=True)
class NeuralSegmentationConfig:
    """Runtime and mask-cleanup settings for v4 inference."""

    batch_size: int = 8
    threshold: float = V4_MODEL.threshold
    expected_spacing_mm: float = V4_MODEL.spacing_mm
    spacing_tolerance_mm: float = 1e-3
    minimum_component_pixels: int = 80
    maximum_hole_pixels: int = 64
    closing_radius_pixels: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        if self.expected_spacing_mm <= 0:
            raise ValueError("expected_spacing_mm must be greater than zero")
        if self.spacing_tolerance_mm < 0:
            raise ValueError("spacing_tolerance_mm must not be negative")
        if self.minimum_component_pixels < 0:
            raise ValueError("minimum_component_pixels must not be negative")
        if self.maximum_hole_pixels < 0:
            raise ValueError("maximum_hole_pixels must not be negative")
        if self.closing_radius_pixels < 0:
            raise ValueError("closing_radius_pixels must not be negative")


@dataclass(frozen=True, slots=True)
class NeuralSegmentationDiagnostics:
    """Provenance and runtime details for one v4 segmentation."""

    model_name: str
    model_version: str
    threshold: float
    provider: str
    bone_voxels: int
    slice_count: int
    inference_seconds: float


class NeuralMaskPostprocessor:
    """Apply the same conservative 2-D cleanup used for v4 preannotations."""

    def __init__(self, config: NeuralSegmentationConfig) -> None:
        self.config = config

    def apply(self, probability: np.ndarray) -> np.ndarray:
        config = self.config
        mask = np.asarray(probability >= config.threshold, dtype=bool)
        if config.closing_radius_pixels:
            mask = binary_closing(
                mask,
                structure=disk(config.closing_radius_pixels),
            )
        components, _ = label(mask)
        component_sizes = np.bincount(components.ravel())
        mask &= component_sizes[components] >= config.minimum_component_pixels
        holes, _ = label(~mask)
        hole_sizes = np.bincount(holes.ravel())
        border_ids = np.unique(
            np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1]))
        )
        fill = hole_sizes[holes] <= config.maximum_hole_pixels
        fill &= ~np.isin(holes, border_ids)
        mask |= fill
        return np.asarray(mask, dtype=bool)


def _create_onnx_session(
    path: Path,
    providers: tuple[str, ...] | None,
) -> InferenceSession:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required for v4 inference. Run `uv sync`."
        ) from exc
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    arguments: dict[str, object] = {"sess_options": options}
    if providers is not None:
        arguments["providers"] = list(providers)
    return ort.InferenceSession(str(path), **arguments)


class V4CBCTBoneSegmenter:
    """Segment a canonical LPS CBCT volume with the published v4 model.

    The model consumes three adjacent axial slices and was trained at 0.8 mm
    isotropic spacing. A session and model are loaded lazily on first use.
    """

    def __init__(
        self,
        config: NeuralSegmentationConfig | None = None,
        *,
        model_manager: ModelManager | None = None,
        model_path: str | Path | None = None,
        offline: bool = False,
        providers: Sequence[str] | None = None,
        download_progress: ProgressCallback | None = None,
        session_factory: SessionFactory | None = None,
        session: InferenceSession | None = None,
    ) -> None:
        if model_path is not None and model_manager is not None:
            raise ValueError("model_path and model_manager are mutually exclusive")
        self.config = config or NeuralSegmentationConfig()
        self.model_manager = model_manager or ModelManager()
        self._model_path = (
            Path(model_path).expanduser().resolve() if model_path is not None else None
        )
        self.offline = bool(offline)
        self.providers = tuple(providers) if providers is not None else None
        self.download_progress = download_progress
        self._session_factory = session_factory or _create_onnx_session
        self._session = session
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._postprocessor = NeuralMaskPostprocessor(self.config)

    @property
    def model_path(self) -> Path:
        """Resolve and return the local model path."""

        if self._model_path is not None:
            if not self._model_path.is_file():
                raise FileNotFoundError(f"model file not found: {self._model_path}")
            return self._model_path
        self._model_path = self.model_manager.get(
            offline=self.offline,
            progress=self.download_progress,
        )
        return self._model_path

    def prepare(self) -> Path:
        """Download, verify, and load the model before the first segmentation."""

        self._ensure_session()
        return self.model_path

    def _ensure_session(self) -> InferenceSession:
        if self._session is None:
            self._session = self._session_factory(self.model_path, self.providers)
        if self._input_name is None:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError("v4 model must have exactly one input and output")
            self._input_name = str(inputs[0].name)
            self._output_name = str(outputs[0].name)
        return self._session

    @staticmethod
    def _normalization(volume: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(volume[np.isfinite(volume)], dtype=np.float32)
        if finite.size == 0:
            raise ValueError("volume contains no finite values")
        stride = max(1, finite.size // 1_000_000)
        low, high = np.percentile(finite[::stride], (0.5, 99.5))
        if high <= low:
            high = low + 1.0
        return float(low), float(high)

    @staticmethod
    def _normalized_plane(plane: np.ndarray, low: float, high: float) -> np.ndarray:
        scaled = (np.asarray(plane, dtype=np.float32) - low) / (high - low)
        return np.nan_to_num(
            np.clip(scaled, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _padded_shape(height: int, width: int) -> tuple[int, int]:
        return int(np.ceil(height / 8) * 8), int(np.ceil(width / 8) * 8)

    def _make_batch(
        self,
        volume: np.ndarray,
        indices: range,
        low: float,
        high: float,
    ) -> np.ndarray:
        height, width = volume.shape[1], volume.shape[0]
        padded_height, padded_width = self._padded_shape(height, width)
        batch = np.zeros(
            (len(indices), 3, padded_height, padded_width),
            dtype=np.float32,
        )
        last = volume.shape[2] - 1
        for row, index in enumerate(indices):
            neighbours = (max(0, index - 1), index, min(last, index + 1))
            for channel, neighbour in enumerate(neighbours):
                plane = self._normalized_plane(
                    volume[:, :, neighbour].T,
                    low,
                    high,
                )
                batch[row, channel, :height, :width] = plane
        return batch

    def _validate_spacing(
        self,
        spacing_mm: float | tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if np.isscalar(spacing_mm):
            spacing = (float(spacing_mm),) * 3
        else:
            spacing = tuple(float(value) for value in spacing_mm)
        if len(spacing) != 3 or not np.all(np.isfinite(spacing)):
            raise ValueError("spacing_mm must contain three finite values")
        if not np.allclose(
            spacing,
            self.config.expected_spacing_mm,
            rtol=0.0,
            atol=self.config.spacing_tolerance_mm,
        ):
            raise ValueError(
                f"v4 expects {self.config.expected_spacing_mm:g} mm isotropic input; "
                f"got {spacing}"
            )
        return spacing

    def segment(
        self,
        volume: np.ndarray,
        spacing_mm: float | tuple[float, float, float] = V4_MODEL.spacing_mm,
    ) -> tuple[np.ndarray, NeuralSegmentationDiagnostics]:
        """Return a full-resolution binary mask for a canonical LPS volume."""

        array = np.asarray(volume, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"volume must be 3-D, got shape {array.shape}")
        if array.size == 0 or array.shape[2] == 0:
            raise ValueError("volume must not be empty")
        self._validate_spacing(spacing_mm)
        low, high = self._normalization(array)
        session = self._ensure_session()
        assert self._input_name is not None and self._output_name is not None
        output = np.zeros(array.shape, dtype=np.uint8)
        height, width = array.shape[1], array.shape[0]
        started = time.perf_counter()
        for start in range(0, array.shape[2], self.config.batch_size):
            indices = range(start, min(array.shape[2], start + self.config.batch_size))
            inputs = self._make_batch(array, indices, low, high)
            probabilities = np.asarray(
                session.run(
                    [self._output_name],
                    {self._input_name: inputs},
                )[0],
                dtype=np.float32,
            )
            expected = (len(indices), 1, inputs.shape[2], inputs.shape[3])
            if probabilities.shape != expected:
                raise RuntimeError(
                    f"v4 model returned shape {probabilities.shape}, "
                    f"expected {expected}"
                )
            for row, index in enumerate(indices):
                plane = probabilities[row, 0, :height, :width]
                output[:, :, index] = self._postprocessor.apply(plane).T
        elapsed = time.perf_counter() - started
        providers = tuple(session.get_providers())
        diagnostics = NeuralSegmentationDiagnostics(
            model_name=V4_MODEL.name,
            model_version=V4_MODEL.version,
            threshold=self.config.threshold,
            provider=providers[0] if providers else "unknown",
            bone_voxels=int(output.sum()),
            slice_count=int(array.shape[2]),
            inference_seconds=float(elapsed),
        )
        return output, diagnostics


__all__ = [
    "InferenceSession",
    "NeuralMaskPostprocessor",
    "NeuralSegmentationConfig",
    "NeuralSegmentationDiagnostics",
    "V4CBCTBoneSegmenter",
]
