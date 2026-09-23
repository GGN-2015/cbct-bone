# cbct-bone

`cbct-bone` is an object-oriented, ready-to-use Python package for extracting binary bone masks from 3D images produced by the target intraoperative CBCT scanner. Its default backend is a v4 2.5D U-Net trained from four manually annotated cases. Users do not need to train a model, configure a network, or download weights manually.

> This version is a research tool, not a medical device. A qualified professional must review every automatic result before it is used for diagnosis, surgical planning, or finite-element analysis.

## Features

- Python 3.12 or later is supported, with [uv](https://docs.astral.sh/uv/) managing the development environment.
- DICOM, NIfTI, and `.ubd.npz` reading is delegated exclusively to `ct_mri_dicom_nii_reader`.
- The package depends on `med-image-seg==0.1.1` from PyPI, so machine-generated annotation JSON can enter the same bilingual GUI and Python annotation workflow. The GUI defaults to English and can switch to Chinese at runtime.
- The default v4 ONNX model is downloaded automatically from GitHub Releases on first use; the file is approximately 4.2 MiB.
- Model downloads support `.part` resume, retries, an interprocess lock, SHA-256 verification, and atomic installation.
- CPU inference uses ONNX Runtime and does not require PyTorch. Other ONNX Runtime execution backends can be selected with the provider option.
- A weight-free 3D adaptive-threshold method remains available as an explicit fallback backend.
- Inputs are normalized to an isotropic LPS grid, and binary masks can be written as NIfTI or `.ubd.npz`.
- `MedicalVolumeReader`, `V4CBCTBoneSegmenter`, `ModelManager`, `MedicalMaskWriter`, `MachineAnnotationWriter`, and `CBCTBoneExtractor` can be reused or replaced independently.
- The annotation GUI lives in the separate `med-image-seg` project, so this segmentation package does not carry GUI implementation or training dependencies.

## Installation

The project requires Python 3.12 or later.

```powershell
uv sync --frozen
```

Install it as a dependency:

```powershell
uv add cbct-bone
```

Before a PyPI release is available, install directly from GitHub:

```powershell
uv add "cbct-bone @ git+https://github.com/GGN-2015/cbct-bone.git"
```

## Command line

The smallest invocation is:

```powershell
uv run cbct-bone path\to\dicom path\to\bone_mask.nii.gz
```

The first call downloads and verifies the v4 weights. Later calls use the local cache. The v4 model requires `0.8 mm` isotropic input, which matches the CLI default.

A single DICOM file, `.nii`, `.nii.gz`, or `.ubd.npz` file is also accepted:

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz --spacing 0.8
```

Print diagnostics as JSON:

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz --json
```

Also write per-slice polygon JSON that can be edited by `med-image-seg`:

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz --annotation-json
```

When `--annotation-json` has no explicit path, NIfTI and UBD inputs use `scan.med-image-seg.json`, while DICOM uses `.med-image-seg.json` in the series directory. A destination can be supplied explicitly:

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz `
  --annotation-json D:\review\scan.med-image-seg.json
```

Machine contours are marked as `prototype` in the JSON. If the target file already exists, the writer replaces old machine contours while preserving `manual` contours, review state, and forced-empty slices. Machine and manual results on the same slice are combined by union in `med-image-seg`. Every inferred slice, including an empty slice, is stored as a keyframe, so reviewing a complete model result does not introduce additional interpolation.

Open the generated annotations directly in the standalone review tool:

```powershell
cd C:\Users\neko\Desktop\github\med-image-seg
uv run med-image-seg --image "D:\images\scan.nii.gz"
```

Use only an already verified cached model when no network is available:

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz --offline
```

Select a local model, cache directory, or ONNX Runtime provider:

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz `
  --model D:\models\cbct-bone-intraoperative-v4.onnx `
  --provider CPUExecutionProvider
```

The weight-free adaptive backend can be selected explicitly and permits a different processing resolution:

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz `
  --backend adaptive --spacing 1.0
```

On Windows, the default model cache is `%LOCALAPPDATA%\cbct-bone\models`; Linux and macOS use their standard user cache directories. Set `CBCT_BONE_CACHE_DIR` or pass `--cache-dir` to override it. The registered model size and SHA-256 are fixed in code, and an invalid file never enters the valid cache.

Every model request reads the standard `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` environment variables, including their lowercase equivalents. For example:

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:10808"
$env:HTTPS_PROXY = "http://127.0.0.1:10808"
uv run cbct-bone scan.nii.gz bone_mask.nii.gz
```

Python's standard library resolves proxy behavior according to the operating system. Proxy addresses and credentials are never written to the model cache.

List all options:

```powershell
uv run cbct-bone --help
```

## Annotation tool

Polygon annotation, automatic interpolation, immediate JSON persistence, and the GUI have moved to the independent [`med-image-seg`](https://github.com/GGN-2015/med-image-seg) project, which is installed as a runtime dependency of `cbct-bone`. `cbct-bone` itself still has no PyTorch or prototype-training dependency.

## Python API

The simplest API reads an image, obtains the v4 weights, runs inference, and writes the mask:

```python
from cbct_bone import segment

result = segment(
    "path/to/dicom",
    "bone_mask.nii.gz",
    annotation_path="path/to/dicom/.med-image-seg.json",
)
print(result.mask.shape)
print(result.diagnostics)
print(result.annotation_path)
```

For composition and dependency injection, use the class interface:

```python
from cbct_bone import (
    CBCTBoneExtractor,
    MedicalMaskWriter,
    MedicalVolumeReader,
    ModelManager,
    NeuralSegmentationConfig,
    V4CBCTBoneSegmenter,
)

reader = MedicalVolumeReader(spacing_mm=0.8)
models = ModelManager()  # download + resume + SHA-256 verification
algorithm = V4CBCTBoneSegmenter(
    NeuralSegmentationConfig(batch_size=8),
    model_manager=models,
    offline=False,
)
extractor = CBCTBoneExtractor(
    reader=reader,
    segmenter=algorithm,
    writer=MedicalMaskWriter(),
)

result = extractor.extract(
    "path/to/dicom",
    "bone_mask.nii.gz",
    annotation_path="path/to/dicom/.med-image-seg.json",
)
print(result.mask.shape)
print(result.diagnostics)
```

Prepare the weights in advance for fully offline inference:

```python
from cbct_bone import ModelManager

model_path = ModelManager().get()
print(model_path)
```

Reuse only the model class when a NumPy LPS volume is already available:

```python
from cbct_bone import V4CBCTBoneSegmenter

mask, diagnostics = V4CBCTBoneSegmenter().segment(
    volume,
    spacing_mm=(0.8, 0.8, 0.8),
)
```

`CBCTBoneExtractor` uses dependency injection. A custom component only needs methods compatible with the `VolumeReader`, `BoneMaskSegmenter`, or `MaskWriter` protocol to replace that stage.

## Algorithm selection

The default v4 model is a 2.5D U-Net tuned for the current target intraoperative CBCT scanner. It uses the current axial slice and its two neighbors, applies robust intensity normalization within each volume, and uses the same threshold and 2D connected-component postprocessing as the training pre-annotation workflow. The training set contains only four cases and 208 manually annotated slices. The model must therefore be treated as a scanner-specific annotation aid, not evidence of cross-scanner or whole-body generalization. See [MODEL_CARD.md](MODEL_CARD.md) for the full limitations.

The `adaptive` backend retains a 3D method without anatomical priors:

1. Three-class multi-Otsu histogram analysis estimates weak-bone and strong-bone thresholds.
2. Local mean and variance in a millimeter-scale 3D neighborhood suppress soft tissue and artifacts that only resemble bone in global intensity.
3. 3D hysteresis region growing expands from strong-bone seeds through weak-bone candidates to recover low-density boundaries affected by partial volume.
4. Physical-scale closing, enclosed-cavity filling, and minimum-component filtering produce the final binary mask.

This fallback follows the adaptive bone-segmentation approach of Zhang et al., combining initial classification, 3D correlation iteration, and region growing with modern multi-Otsu thresholding and 3D morphology. It requires neither training data nor model weights, but it has no learned anatomical semantics.

For scanners with unusual intensity distributions, thresholds can be overridden explicitly:

```powershell
uv run cbct-bone scan.nii.gz mask.nii.gz --backend adaptive `
  --weak-threshold 180 --strong-threshold 450
```

The weak threshold must be lower than the strong threshold. Inspect the automatically selected values with `--json`, then establish scanner-level settings from a small reviewed dataset.

## I/O and coordinate conventions

- Input paths are read only through the public loader in `ct_mri_dicom_nii_reader`; this project contains no second medical-image reader.
- Inputs are resampled to an isotropic `(L, P, S)` array before processing.
- SimpleITK is used to write NIfTI, but never to read source images.
- NIfTI output retains the resampled grid spacing and LPS origin, with an identity direction matrix.
- `.ubd.npz` does not preserve a physical origin. It is useful for fast internal exchange, but not for cross-application workflows that depend on patient coordinates.

## Validation status

The v4 model was trained from 208 manually annotated slices in four CBCT studies acquired on the target scanner. Of those slices, 165 were used for training and 43 spatially held-out slices for internal validation. Internal Dice was `0.8678`, precision was `0.8493`, and recall was `0.8872`. Adjacent slices still come from the same patients, so these values are not an independent patient-level test and do not estimate real clinical generalization.

Automated tests cover model caching, resumed downloads, SHA-256 failure, offline mode, ONNX axis restoration, spacing constraints, NIfTI geometry round trips, CLI behavior, and object-oriented dependency injection. Numerical comparison between the ONNX export and the original PyTorch v4 model found a maximum absolute error of `1.58e-5`. Synthetic tests prevent algorithm and geometry regressions; they are not clinical validation.

Before use on real scanners, build a manually labeled test set stratified by scanner, anatomy, dose protocol, and metal implants. Report Dice, surface Dice, HD95, sensitivity, and false-positive volume.

Multicenter testing on open SynthRAD2023 pelvic CBCT data showed a substantial scanner-domain shift for the pure-threshold method. The v4 model has likewise not been validated on independent patients, other scanners, knees, or whole-body anatomy. The current automatic algorithm is not clinically validated; future evaluation requires a patient-separated independent test set and the same volumetric and surface metrics.

## Model release

- Release: `model-v4.0.0`
- File: `cbct-bone-intraoperative-v4.onnx`
- Size: `4,428,600` bytes
- SHA-256: `f18ed7c9071b35c613ff748088bfadbd5bd7691793c1c8b80bf9013315c7abc6`
- Training grid: `0.8 mm` isotropic LPS
- Segmentation threshold: `0.775`

The runtime accepts a model file only when both its registered size and SHA-256 match.

Run the quality checks:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=cbct_bone
uv build
```

## References

- Zhang J, Yan C-H, Chui C-K, Ong S-H. [Fast segmentation of bone in CT images using 3D adaptive thresholding](https://doi.org/10.1016/j.compbiomed.2009.11.020). Computers in Biology and Medicine. 2010;40(2):231-236.
- van Eijnatten M, et al. [CT image segmentation methods for bone used in medical additive manufacturing](https://doi.org/10.1016/j.medengphy.2017.10.008). Medical Engineering & Physics. 2018;51:6-16.
- Isensee F, et al. [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z). Nature Methods. 2021;18:203-211.
- Dot G, et al. [DentalSegmentator: robust open source deep learning-based CT and CBCT image segmentation](https://doi.org/10.1016/j.jdent.2024.105130). Journal of Dentistry. 2024;147:105130.

## License

[MIT](LICENSE)
