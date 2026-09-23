"""Versioned model discovery, verified caching, and resumable downloads."""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable metadata for one published inference model."""

    name: str
    version: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    threshold: float
    spacing_mm: float

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        int(self.sha256, 16)
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be greater than zero")


V4_MODEL = ModelSpec(
    name="cbct-bone-intraoperative-v4",
    version="4.0.0",
    filename="cbct-bone-intraoperative-v4.onnx",
    url=(
        "https://github.com/GGN-2015/cbct-bone/releases/download/"
        "model-v4.0.0/cbct-bone-intraoperative-v4.onnx"
    ),
    sha256="f18ed7c9071b35c613ff748088bfadbd5bd7691793c1c8b80bf9013315c7abc6",
    size_bytes=4_428_600,
    threshold=0.775,
    spacing_mm=0.8,
)

V5_MODEL = ModelSpec(
    name="cbct-bone-intraoperative-v5",
    version="5.0.0",
    filename="cbct-bone-intraoperative-v5.onnx",
    url=(
        "https://github.com/GGN-2015/cbct-bone/releases/download/"
        "model-v5.0.0/cbct-bone-intraoperative-v5.onnx"
    ),
    sha256="e0e5711b856fd22fbb9e0dd31d49c3e73d7b4ea15988029e5acff626304f32f7",
    size_bytes=4_428_425,
    threshold=0.8,
    spacing_mm=0.8,
)

LATEST_MODEL = V5_MODEL


class ModelError(RuntimeError):
    """Base exception for model acquisition failures."""


class ModelUnavailableError(ModelError):
    """Raised when a requested model is not available locally or remotely."""


class ModelIntegrityError(ModelError):
    """Raised when a model does not match its published size or checksum."""


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """Progress information reported while a model is downloaded."""

    downloaded_bytes: int
    total_bytes: int


ProgressCallback = Callable[[DownloadProgress], None]


class _Response(Protocol):
    status: int
    headers: object

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> None: ...


OpenUrl = Callable[..., _Response]


def _open_with_environment_proxy(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> _Response:
    """Open a URL using the current process proxy environment.

    ``urllib.request.getproxies`` reads ``HTTP_PROXY``, ``HTTPS_PROXY`` and
    ``NO_PROXY`` (including their lowercase variants) using platform-native
    rules. Resolving it for every request means applications may set these
    variables before calling :meth:`ModelManager.get` without importing this
    module again.
    """

    proxies = urllib.request.getproxies()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    return opener.open(request, timeout=timeout)


def default_model_cache() -> Path:
    """Return the platform-appropriate model cache directory."""

    override = os.environ.get("CBCT_BONE_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return (Path(base) / "cbct-bone" / "models").resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (Path(xdg_cache) / "cbct-bone" / "models").resolve()
    return (Path.home() / ".cache" / "cbct-bone" / "models").resolve()


class ModelManager:
    """Resolve a model from a verified local cache or its release asset.

    Interrupted downloads remain in a ``.part`` file. A later call requests
    the remaining byte range and atomically promotes the file only after its
    size and SHA-256 digest match :class:`ModelSpec`.
    """

    def __init__(
        self,
        spec: ModelSpec = LATEST_MODEL,
        *,
        cache_dir: str | Path | None = None,
        opener: OpenUrl | None = None,
        timeout_seconds: float = 60.0,
        retries: int = 3,
        lock_timeout_seconds: float = 120.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if retries < 0:
            raise ValueError("retries must not be negative")
        self.spec = spec
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else default_model_cache()
        )
        self._opener = opener or _open_with_environment_proxy
        self.timeout_seconds = float(timeout_seconds)
        self.retries = int(retries)
        self.lock_timeout_seconds = float(lock_timeout_seconds)

    @property
    def path(self) -> Path:
        return self.cache_dir / self.spec.filename

    @property
    def partial_path(self) -> Path:
        return self.cache_dir / f"{self.spec.filename}.part"

    @property
    def lock_path(self) -> Path:
        return self.cache_dir / f"{self.spec.filename}.lock"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def verify(self, path: str | Path | None = None) -> bool:
        """Return whether ``path`` exactly matches the registered model."""

        candidate = Path(path).expanduser().resolve() if path else self.path
        return (
            candidate.is_file()
            and candidate.stat().st_size == self.spec.size_bytes
            and self._sha256(candidate) == self.spec.sha256
        )

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout_seconds
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(descriptor, str(os.getpid()).encode("ascii"))
            except FileExistsError as exc:
                try:
                    stale = time.time() - self.lock_path.stat().st_mtime > 3600
                except FileNotFoundError:
                    continue
                if stale:
                    self.lock_path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    raise ModelUnavailableError(
                        f"timed out waiting for model cache lock: {self.lock_path}"
                    ) from exc
                time.sleep(0.1)
        try:
            yield
        finally:
            os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)

    @staticmethod
    def _content_range_total(headers: object) -> int | None:
        get = getattr(headers, "get", None)
        if get is None:
            return None
        value = get("Content-Range")
        if not value or "/" not in value:
            return None
        total = str(value).rsplit("/", 1)[-1]
        return int(total) if total.isdigit() else None

    def _download_once(
        self,
        *,
        progress: ProgressCallback | None,
    ) -> None:
        partial = self.partial_path
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset == self.spec.size_bytes:
            if self.verify(partial):
                return
            partial.unlink()
            offset = 0
        elif offset > self.spec.size_bytes:
            partial.unlink()
            offset = 0
        headers = {"User-Agent": "cbct-bone/0.3.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(self.spec.url, headers=headers)
        with self._opener(request, timeout=self.timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            append = offset > 0 and status == 206
            if status not in (200, 206):
                raise ModelUnavailableError(
                    f"model server returned HTTP status {status}"
                )
            if status == 206 and not offset:
                raise ModelUnavailableError("model server returned an unexpected range")
            if offset and not append:
                offset = 0
            total = self._content_range_total(response.headers) or self.spec.size_bytes
            mode = "ab" if append else "wb"
            downloaded = offset
            with partial.open(mode) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(DownloadProgress(downloaded, total))
                output.flush()
                os.fsync(output.fileno())

    def _download(self, *, progress: ProgressCallback | None) -> None:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self._download_once(progress=progress)
                return
            except (OSError, urllib.error.URLError, ModelUnavailableError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(min(2**attempt, 4))
        raise ModelUnavailableError(
            f"failed to download {self.spec.name} from {self.spec.url}: {last_error}"
        ) from last_error

    def get(
        self,
        *,
        offline: bool = False,
        force_download: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Return a verified model path, downloading it when necessary."""

        if not force_download and self.verify():
            return self.path
        if offline:
            raise ModelUnavailableError(
                f"model is not available in offline cache: {self.path}"
            )
        with self._exclusive_lock():
            if not force_download and self.verify():
                return self.path
            if force_download:
                self.path.unlink(missing_ok=True)
                self.partial_path.unlink(missing_ok=True)
            elif self.path.exists():
                self.path.unlink()
            self._download(progress=progress)
            if not self.verify(self.partial_path):
                actual_size = (
                    self.partial_path.stat().st_size
                    if self.partial_path.exists()
                    else 0
                )
                self.partial_path.unlink(missing_ok=True)
                raise ModelIntegrityError(
                    f"downloaded model failed integrity verification "
                    f"(size={actual_size}, expected={self.spec.size_bytes})"
                )
            os.replace(self.partial_path, self.path)
            return self.path

    def clear(self) -> None:
        """Remove this model and its partial download from the cache."""

        self.path.unlink(missing_ok=True)
        self.partial_path.unlink(missing_ok=True)


__all__ = [
    "LATEST_MODEL",
    "V4_MODEL",
    "V5_MODEL",
    "DownloadProgress",
    "ModelError",
    "ModelIntegrityError",
    "ModelManager",
    "ModelSpec",
    "ModelUnavailableError",
    "ProgressCallback",
    "default_model_cache",
]
