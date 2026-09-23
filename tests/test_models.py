from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.request import Request

import pytest

from cbct_bone import models as model_module
from cbct_bone.models import (
    ModelIntegrityError,
    ModelManager,
    ModelSpec,
    ModelUnavailableError,
)


class FakeResponse(BytesIO):
    def __init__(self, data: bytes, *, status: int, headers: dict[str, str]) -> None:
        super().__init__(data)
        self.status = status
        self.headers = headers

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class InterruptingResponse(FakeResponse):
    def __init__(self, data: bytes, *, first_chunk: int) -> None:
        super().__init__(data, status=200, headers={})
        self.first_chunk = first_chunk
        self.interrupted = False

    def read(self, size: int = -1) -> bytes:
        if self.tell() == 0:
            return super().read(self.first_chunk)
        if not self.interrupted:
            self.interrupted = True
            raise TimeoutError("simulated connection drop")
        return super().read(size)


def _spec(data: bytes) -> ModelSpec:
    return ModelSpec(
        name="test-model",
        version="1.0.0",
        filename="test.onnx",
        url="https://example.invalid/test.onnx",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        threshold=0.5,
        spacing_mm=0.8,
    )


def test_model_manager_resumes_partial_download(tmp_path: Path) -> None:
    data = b"verified model bytes" * 100
    spec = _spec(data)
    ranges: list[str | None] = []

    def opener(request: Request, **_kwargs: object) -> FakeResponse:
        value = request.get_header("Range")
        ranges.append(value)
        offset = int(value.removeprefix("bytes=").removesuffix("-")) if value else 0
        return FakeResponse(
            data[offset:],
            status=206 if offset else 200,
            headers={"Content-Range": f"bytes {offset}-{len(data) - 1}/{len(data)}"},
        )

    manager = ModelManager(spec, cache_dir=tmp_path, opener=opener)
    manager.cache_dir.mkdir(parents=True, exist_ok=True)
    manager.partial_path.write_bytes(data[:317])

    path = manager.get()

    assert path.read_bytes() == data
    assert ranges == ["bytes=317-"]
    assert not manager.partial_path.exists()
    assert manager.verify()


def test_model_manager_retries_from_interrupted_byte_range(tmp_path: Path) -> None:
    data = b"network payload" * 200
    spec = _spec(data)
    ranges: list[str | None] = []

    def opener(request: Request, **_kwargs: object) -> FakeResponse:
        value = request.get_header("Range")
        ranges.append(value)
        if len(ranges) == 1:
            return InterruptingResponse(data, first_chunk=311)
        assert value == "bytes=311-"
        return FakeResponse(
            data[311:],
            status=206,
            headers={"Content-Range": f"bytes 311-{len(data) - 1}/{len(data)}"},
        )

    manager = ModelManager(spec, cache_dir=tmp_path, opener=opener, retries=1)

    assert manager.get().read_bytes() == data
    assert ranges == [None, "bytes=311-"]


def test_model_manager_uses_verified_cache_without_network(tmp_path: Path) -> None:
    data = b"cached model"
    spec = _spec(data)

    def fail_if_called(*_args: object, **_kwargs: object) -> FakeResponse:
        raise AssertionError("network should not be used")

    manager = ModelManager(spec, cache_dir=tmp_path, opener=fail_if_called)
    manager.cache_dir.mkdir(parents=True, exist_ok=True)
    manager.path.write_bytes(data)

    assert manager.get() == manager.path


def test_model_manager_promotes_complete_partial_without_network(
    tmp_path: Path,
) -> None:
    data = b"completed partial model"
    spec = _spec(data)

    def fail_if_called(*_args: object, **_kwargs: object) -> FakeResponse:
        raise AssertionError("network should not be used")

    manager = ModelManager(spec, cache_dir=tmp_path, opener=fail_if_called)
    manager.partial_path.write_bytes(data)

    assert manager.get().read_bytes() == data
    assert not manager.partial_path.exists()


def test_model_manager_rejects_bad_download(tmp_path: Path) -> None:
    expected = b"expected model"
    bad = b"tampered model"
    spec = _spec(expected)

    def opener(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse(bad, status=200, headers={})

    manager = ModelManager(spec, cache_dir=tmp_path, opener=opener, retries=0)

    with pytest.raises(ModelIntegrityError, match="integrity"):
        manager.get()

    assert not manager.path.exists()
    assert not manager.partial_path.exists()


def test_model_manager_offline_requires_verified_cache(tmp_path: Path) -> None:
    manager = ModelManager(_spec(b"model"), cache_dir=tmp_path)

    with pytest.raises(ModelUnavailableError, match="offline"):
        manager.get(offline=True)


def test_default_opener_reads_http_and_https_proxy_for_each_request() -> None:
    response = FakeResponse(b"model", status=200, headers={})
    opener = Mock()
    opener.open.return_value = response
    proxies = {
        "http": "http://127.0.0.1:10808",
        "https": "http://127.0.0.1:10808",
        "no": "localhost,127.0.0.1",
    }
    request = Request("https://example.invalid/model.onnx")

    with (
        patch("urllib.request.getproxies", return_value=proxies) as getproxies,
        patch("urllib.request.build_opener", return_value=opener) as build_opener,
    ):
        actual = model_module._open_with_environment_proxy(request, timeout=12.5)

    assert actual is response
    getproxies.assert_called_once_with()
    handler = build_opener.call_args.args[0]
    assert handler.proxies == proxies
    opener.open.assert_called_once_with(request, timeout=12.5)
