# cbct-bone intraoperative v4 model card

## Model details

- Model ID: `cbct-bone-intraoperative-v4`
- Version: `4.0.0`
- Architecture: compact 2.5D U-Net, three adjacent axial slices
- Runtime format: ONNX, opset 18
- Input: one canonical LPS CBCT volume resampled to 0.8 mm isotropic spacing
- Output: per-slice bone probability, thresholded at 0.775
- License: MIT

The model is intended to accelerate polygon correction and produce research-use
bone masks for the same intraoperative CBCT device represented in its training
data. It is not a medical device.

## Training data

The model was trained from four locally annotated, anonymized CBCT studies:

- 208 trusted manual slices in total;
- 165 training slices;
- 43 spatially held-out validation slices.

No source DICOM files, identity fields, paths, or polygon projects are included
in the released model asset. The weights are nevertheless derived from medical
data and should be distributed only under the data owner's authorization.

## Internal validation

- Dice: 0.8678
- Precision: 0.8493
- Recall: 0.8872

These metrics use held-out slices from the same four studies. Adjacent slices
are correlated, so this is a development metric rather than an independent
patient-level evaluation.

## Limitations

- Only one target CBCT device/domain is represented.
- Four studies are insufficient to claim whole-body, cross-device, or robust
  anatomy coverage.
- Independent knee, pelvis, metal implant, low-dose, fracture, and outlier
  cohorts have not been evaluated.
- The model can omit low-contrast cancellous bone or include metal and dense
  soft-tissue artifacts.
- Every result requires professional review before clinical, surgical,
  manufacturing, or finite-element use.

## Release integrity

- Asset: `cbct-bone-intraoperative-v4.onnx`
- Size: 4,428,600 bytes
- SHA-256: `f18ed7c9071b35c613ff748088bfadbd5bd7691793c1c8b80bf9013315c7abc6`
- PyTorch-to-ONNX parity: maximum absolute error 1.58e-5 on a deterministic
  two-sample input batch.

`ModelManager` verifies both size and SHA-256 before the asset enters the valid
cache.
