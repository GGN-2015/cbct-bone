"""Write segmentation masks as editable med-image-seg polygon JSON."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from skimage.measure import approximate_polygon, find_contours

from .io import MedicalVolume

_IGNORED_DICOM_SUFFIXES = {
    ".avi",
    ".bak",
    ".bmp",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".json",
    ".mp4",
    ".nii",
    ".npz",
    ".png",
    ".tmp",
}


def _strip_medical_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in (".nii.gz", ".ubd.npz", ".nii"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _canonical_source(path: str | Path) -> tuple[Path, bool]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"image source not found: {source}")
    lower = source.name.lower()
    if source.is_dir() or lower.endswith(".dcm") or not source.suffix:
        return (source if source.is_dir() else source.parent), True
    if lower.endswith((".nii", ".nii.gz", ".ubd.npz")):
        return source, False
    raise ValueError(
        "unsupported image source; expected a DICOM directory/file, "
        ".nii, .nii.gz, or .ubd.npz"
    )


def _dicom_candidates(directory: Path) -> list[Path]:
    files = sorted(
        (
            item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() not in _IGNORED_DICOM_SUFFIXES
        ),
        key=lambda item: item.name,
    )
    if not files:
        raise ValueError(f"DICOM directory contains no candidate files: {directory}")
    return files


def source_fingerprint(path: str | Path) -> str:
    """Return the content fingerprint used by med-image-seg sidecar files."""

    source, is_dicom = _canonical_source(path)
    digest = hashlib.sha256()
    digest.update(("dicom" if is_dicom else _source_kind(source)).encode("ascii"))
    if not is_dicom:
        stat = source.stat()
        digest.update(source.name.encode("utf-8", errors="surrogatepass"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        with source.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
        return digest.hexdigest()

    files = _dicom_candidates(source)
    digest.update(str(len(files)).encode("ascii"))
    for item in files:
        stat = item.stat()
        digest.update(item.name.encode("utf-8", errors="surrogatepass"))
        digest.update(str(stat.st_size).encode("ascii"))
    for item in dict.fromkeys((files[0], files[len(files) // 2], files[-1])):
        with item.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def _source_kind(path: Path) -> str:
    return "ubd" if path.name.lower().endswith(".ubd.npz") else "nifti"


def annotation_path_for(path: str | Path) -> Path:
    """Return the sidecar location automatically detected by med-image-seg."""

    source, is_dicom = _canonical_source(path)
    if is_dicom:
        return source / ".med-image-seg.json"
    base = _strip_medical_suffix(source.name)
    return source.parent / f"{base}.med-image-seg.json"


def _signed_area(points: list[tuple[float, float]]) -> float:
    array = np.asarray(points, dtype=np.float64)
    x = array[:, 0]
    y = array[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class MachineAnnotationWriter:
    """Convert a binary LPS mask to editable prototype polygons."""

    def __init__(self, contour_tolerance: float = 0.75) -> None:
        if not np.isfinite(contour_tolerance) or contour_tolerance < 0:
            raise ValueError("contour_tolerance must be finite and non-negative")
        self.contour_tolerance = float(contour_tolerance)

    def _plane_polygons(self, plane: np.ndarray) -> list[dict[str, object]]:
        padded = np.pad(np.asarray(plane, dtype=np.uint8), 1)
        contours = find_contours(
            padded,
            0.5,
            fully_connected="high",
            positive_orientation="low",
        )
        polygons: list[tuple[float, dict[str, object]]] = []
        height, width = plane.shape
        for contour in contours:
            simplified = approximate_polygon(contour, self.contour_tolerance)
            points = [
                (
                    float(np.clip(point[1] - 1.0, 0.0, width - 1.0)),
                    float(np.clip(point[0] - 1.0, 0.0, height - 1.0)),
                )
                for point in simplified
            ]
            if len(points) > 1 and np.allclose(points[0], points[-1]):
                points.pop()
            points = [
                point
                for index, point in enumerate(points)
                if index == 0 or not np.allclose(point, points[index - 1])
            ]
            if len(points) < 3:
                continue
            area = _signed_area(points)
            if abs(area) < 1e-6:
                continue
            polygons.append(
                (
                    abs(area),
                    {
                        "points": points,
                        "operation": "add" if area > 0 else "erase",
                        "source": "prototype",
                    },
                )
            )
        polygons.sort(key=lambda item: item[0], reverse=True)
        return [polygon for _, polygon in polygons]

    @staticmethod
    def _load_existing(path: Path) -> dict[str, object] | None:
        for candidate in (path, Path(f"{path}.bak")):
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _validate_existing(
        payload: dict[str, object],
        mask: np.ndarray,
        volume: MedicalVolume,
        fingerprint: str,
    ) -> None:
        try:
            shape = tuple(payload["shape_lps"])
            spacing = payload["spacing_lps_mm"]
            origin = payload["origin_lps_mm"]
            stored_fingerprint = payload["source_fingerprint"]
            slices = payload.get("slices", {})
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "existing annotation JSON has an incompatible schema"
            ) from exc
        if not isinstance(slices, dict):
            raise ValueError("existing annotation slices must be a JSON object")
        if shape != tuple(mask.shape):
            raise ValueError("existing annotation shape does not match segmentation")
        if not np.allclose(spacing, volume.spacing_mm):
            raise ValueError("existing annotation spacing does not match segmentation")
        if not np.allclose(origin, volume.origin_lps_mm):
            raise ValueError("existing annotation origin does not match segmentation")
        if stored_fingerprint != fingerprint:
            raise ValueError("existing annotation fingerprint does not match image")

    @staticmethod
    def _atomic_save(path: Path, payload: dict[str, object]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{path}.tmp")
        backup = Path(f"{path}.bak")
        backup_temporary = Path(f"{backup}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_file():
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError):
                    pass
                else:
                    shutil.copy2(path, backup_temporary)
                    os.replace(backup_temporary, backup)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
            backup_temporary.unlink(missing_ok=True)
        return path

    def write(
        self,
        path: str | Path,
        mask: np.ndarray,
        reference: MedicalVolume,
        *,
        source_path: str | Path | None = None,
    ) -> Path:
        """Write machine polygons and preserve any existing manual polygons."""

        output = Path(path).expanduser().resolve()
        array = np.asarray(mask, dtype=np.uint8)
        if array.ndim != 3 or array.shape != reference.data.shape:
            raise ValueError("mask must match the 3-D reference volume shape")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("mask must contain only 0 and 1")
        source = reference.source if source_path is None else source_path
        fingerprint = source_fingerprint(source)
        existing = self._load_existing(output)
        if existing is not None:
            self._validate_existing(existing, array, reference, fingerprint)

        existing_slices = {} if existing is None else existing.get("slices", {})
        edit_counter = 0 if existing is None else int(existing.get("edit_counter", 0))
        slices: dict[str, object] = {}
        for index in range(array.shape[2]):
            previous = existing_slices.get(str(index), {})
            if not isinstance(previous, dict):
                raise ValueError("existing slice annotation must be a JSON object")
            manual = [
                polygon
                for polygon in previous.get("polygons", [])
                if polygon.get("source", "manual") != "prototype"
            ]
            machine = self._plane_polygons(array[:, :, index].T)
            for polygon in machine:
                edit_counter += 1
                polygon["edit_order"] = edit_counter
            slices[str(index)] = {
                "completed": bool(previous.get("completed", False)),
                "keyframe": True,
                "force_empty": bool(previous.get("force_empty", False)),
                "polygons": [*machine, *manual],
            }

        now = datetime.now(UTC).isoformat()
        metadata = {} if existing is None else dict(existing.get("metadata", {}))
        metadata["machine_segmentation"] = {
            "producer": "cbct-bone",
            "generated_at": now,
            "contour_tolerance_pixels": self.contour_tolerance,
        }
        payload = {
            "schema_version": 4,
            "image_id": f"IMAGE-{fingerprint[:12].upper()}",
            "source_fingerprint": fingerprint,
            "shape_lps": tuple(int(value) for value in array.shape),
            "spacing_lps_mm": reference.spacing_mm,
            "origin_lps_mm": reference.origin_lps_mm,
            "updated_at": now,
            "edit_counter": edit_counter,
            "metadata": metadata,
            "slices": slices,
        }
        return self._atomic_save(output, payload)


__all__ = [
    "MachineAnnotationWriter",
    "annotation_path_for",
    "source_fingerprint",
]
