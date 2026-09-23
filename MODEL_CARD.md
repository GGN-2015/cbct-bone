# cbct-bone intraoperative v5 model card

## Model details

- Model ID: `cbct-bone-intraoperative-v5`
- Version: `5.0.0`
- Architecture: compact 2.5D U-Net with 1,096,609 parameters
- Input context: three adjacent axial slices
- Runtime format: ONNX, opset 17
- Input: one canonical LPS CBCT volume resampled to 0.8 mm isotropic spacing
- Output: per-slice bone probability, thresholded at 0.8
- License: MIT

The model is intended to accelerate polygon correction and produce research-use
bone masks for the same intraoperative CBCT device represented in its training
data. It is not a medical device.

## Training data

Five locally annotated CBCT studies from the target device were used. Polygon
keyframes were converted to dense masks with the same signed-distance
interpolation implemented by `med-image-seg`. This produced 1,206 axial slices:

- case contributions: 227, 246, 245, 242, and 246 slices;
- model selection: 965 training slices and 241 contiguous validation slices;
- final fit: all 1,206 keyframe and interpolation-derived slices for 58 epochs.

The validation block location was varied across the five studies so that the
aggregate selection set covered different superior-inferior regions. The final
release fit used every labeled and interpolated slice, as requested.

No source DICOM files, identity fields, paths, or polygon projects are included
in the released model asset. The weights are nevertheless derived from medical
data and should be distributed only under the data owner's authorization.

## Internal validation

The model-selection checkpoint achieved:

- Dice: 0.8733
- Precision: 0.8735
- Recall: 0.8731
- Selected threshold: 0.8
- Best epoch: 58

A full-volume sanity check of the final ONNX model on the fifth training project
achieved Dice 0.9280, precision 0.8967, and recall 0.9616. This is a training-case
consistency check, not an independent performance estimate.

All validation slices come from patients represented in training, and most dense
labels are interpolation-derived. The metrics therefore measure development
consistency and cannot establish patient-level, cross-scanner, or clinical
generalization.

## Limitations

- Only one target CBCT device/domain is represented.
- Five studies are insufficient to claim whole-body, cross-device, or robust
  anatomy coverage.
- Interpolated masks inherit errors from their neighboring polygon keyframes and
  are not equivalent to independently traced slice labels.
- Independent knee, pelvis, metal implant, low-dose, fracture, and outlier
  cohorts have not been evaluated.
- The model can omit low-contrast cancellous bone or include metal and dense
  soft-tissue artifacts.
- Every result requires professional review before clinical, surgical,
  manufacturing, or finite-element use.

## Release integrity

- Release: `model-v5.0.0`
- Asset: `cbct-bone-intraoperative-v5.onnx`
- Size: 4,428,425 bytes
- SHA-256: `e0e5711b856fd22fbb9e0dd31d49c3e73d7b4ea15988029e5acff626304f32f7`
- PyTorch-to-ONNX probability parity: maximum absolute error 1.49e-5 on a
  deterministic input.

`ModelManager` verifies both size and SHA-256 before the asset enters the valid
cache. The previous v4 asset remains available through the explicit `v4`
backend for reproducibility.
