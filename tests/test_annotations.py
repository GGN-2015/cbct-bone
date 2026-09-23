from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage.draw import polygon as rasterize_polygon

from cbct_bone.annotations import MachineAnnotationWriter, annotation_path_for
from cbct_bone.io import MedicalVolume


def _volume(source: Path, shape: tuple[int, int, int] = (32, 30, 4)) -> MedicalVolume:
    return MedicalVolume(
        data=np.zeros(shape, dtype=np.float32),
        spacing_mm=(0.8, 0.8, 0.8),
        origin_lps_mm=(1.0, 2.0, 3.0),
        metadata={"modality": "CBCT"},
        source=source,
    )


def _rasterize_payload(payload: dict) -> np.ndarray:
    shape = tuple(payload["shape_lps"])
    mask = np.zeros(shape, dtype=np.uint8)
    for index, annotation in payload["slices"].items():
        plane = np.zeros((shape[1], shape[0]), dtype=bool)
        for item in annotation["polygons"]:
            points = np.asarray(item["points"], dtype=np.float64)
            rows, columns = rasterize_polygon(
                points[:, 1],
                points[:, 0],
                shape=plane.shape,
            )
            plane[rows, columns] = item["operation"] == "add"
        mask[:, :, int(index)] = plane.T
    return mask


def test_machine_mask_is_written_as_editable_prototype_polygons(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scan.nii.gz"
    source.write_bytes(b"source-image")
    volume = _volume(source)
    mask = np.zeros(volume.data.shape, dtype=np.uint8)
    mask[4:18, 5:24, 0] = 1
    mask[5:27, 4:26, 1] = 1
    mask[11:20, 10:19, 1] = 0
    mask[2:10, 3:12, 3] = 1
    mask[20:29, 17:27, 3] = 1

    output = MachineAnnotationWriter(contour_tolerance=0.0).write(
        annotation_path_for(source),
        mask,
        volume,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    reconstructed = _rasterize_payload(payload)
    intersection = np.count_nonzero(mask & reconstructed)
    dice = 2 * intersection / (mask.sum() + reconstructed.sum())

    assert payload["schema_version"] == 4
    assert set(payload["slices"]) == {"0", "1", "2", "3"}
    assert all(item["keyframe"] for item in payload["slices"].values())
    assert all(
        polygon["source"] == "prototype"
        for item in payload["slices"].values()
        for polygon in item["polygons"]
    )
    assert any(
        polygon["operation"] == "erase"
        for polygon in payload["slices"]["1"]["polygons"]
    )
    assert dice > 0.97


def test_rewriting_machine_result_preserves_manual_content(tmp_path: Path) -> None:
    source = tmp_path / "scan.nii.gz"
    source.write_bytes(b"source-image")
    volume = _volume(source)
    mask = np.zeros(volume.data.shape, dtype=np.uint8)
    mask[4:18, 5:24, 0] = 1
    output = annotation_path_for(source)
    writer = MachineAnnotationWriter(contour_tolerance=0.0)
    writer.write(output, mask, volume)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["slices"]["0"]["force_empty"] = True
    payload["slices"]["2"]["polygons"].append(
        {
            "points": [[2, 2], [9, 2], [9, 9], [2, 9]],
            "operation": "add",
            "source": "manual",
            "edit_order": payload["edit_counter"] + 1,
        }
    )
    payload["edit_counter"] += 1
    output.write_text(json.dumps(payload), encoding="utf-8")

    replacement = np.zeros_like(mask)
    replacement[18:28, 12:25, 0] = 1
    writer.write(output, replacement, volume)
    merged = json.loads(output.read_text(encoding="utf-8"))

    assert Path(f"{output}.bak").is_file()
    assert merged["slices"]["0"]["force_empty"] is True
    assert (
        sum(
            polygon["source"] == "manual"
            for polygon in merged["slices"]["2"]["polygons"]
        )
        == 1
    )
    assert all(
        polygon["source"] != "prototype"
        for polygon in merged["slices"]["2"]["polygons"]
    )
