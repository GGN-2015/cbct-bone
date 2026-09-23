# cbct-bone

一个开箱即用、面向对象的 Python 包，用于从目标术中 CBCT 设备的三维影像中提取二值骨骼掩码。默认后端是由四例人工标注数据训练的 v4 2.5D U-Net；用户不需要训练模型、配置网络或手工下载权重。

> 当前版本是研究工具，不是医疗器械。自动结果在诊断、手术规划或有限元分析前必须由专业人员复核。

## 特点

- Python 固定为 3.12，使用 [uv](https://docs.astral.sh/uv/) 管理和锁定环境。
- DICOM、NIfTI 和 `.ubd.npz` 的读取全部委托给 `ct_mri_dicom_nii_reader==0.1.6`。
- 默认使用 v4 ONNX 模型；首次调用自动从 GitHub Release 下载约 4.2 MiB 权重。
- 下载支持 `.part` 断点续传、重试、并发锁、SHA-256 校验和原子落盘。
- CPU 推理依赖 ONNX Runtime，不要求安装 PyTorch；可通过 provider 参数使用其他 ONNX Runtime 执行后端。
- 保留无需权重的 3D 自适应阈值算法作为显式后备后端。
- 输入统一到各向同性 LPS 网格，输出 NIfTI 或 `.ubd.npz` 二值掩码。
- `MedicalVolumeReader`、`V4CBCTBoneSegmenter`、`ModelManager`、
  `MedicalMaskWriter`、`MachineAnnotationWriter` 和 `CBCTBoneExtractor`
  可独立复用或替换。
- 标注 GUI 已拆分到独立的 `med-image-seg` 项目，分割包不再引入 GUI 或训练依赖。

## 安装

项目要求 Python `>=3.12,<3.13`，仓库中的 `.python-version` 已固定到 3.12。

```powershell
uv sync --frozen
```

作为依赖安装：

```powershell
uv add cbct-bone
```

在发布到 PyPI 前，可以从 Git 仓库安装：

```powershell
uv add "cbct-bone @ git+https://github.com/GGN-2015/cbct-bone.git"
```

## 命令行

最简单的调用：

```powershell
uv run cbct-bone path\to\dicom path\to\bone_mask.nii.gz
```

第一次调用会自动下载并校验 v4 权重，后续直接使用本地缓存。v4 模型固定使用 `0.8 mm` 各向同性输入；CLI 默认值已经与之匹配。

也可以输入单个 DICOM 文件、`.nii`、`.nii.gz` 或 `.ubd.npz`：

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz --spacing 0.8
```

输出 JSON 诊断信息：

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz --json
```

同时输出可由 `med-image-seg` 继续修改的逐层多边形 JSON：

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz --annotation-json
```

省略 `--annotation-json` 的路径时，NIfTI/UBD 使用
`scan.med-image-seg.json`，DICOM 使用序列目录内的 `.med-image-seg.json`；
也可以显式指定路径：

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz `
  --annotation-json D:\review\scan.med-image-seg.json
```

机器轮廓在 JSON 中标记为 `prototype`。如果目标 JSON 已存在，程序会替换
旧机器轮廓并保留 `manual` 人工轮廓、复核状态和强制空层；同层机器与人工
结果在 `med-image-seg` 中按并集处理。每个机器推理层（包括空层）都会写为
关键层，因而检查完整模型结果时不会额外产生插值。

生成后可在独立标注工具中直接检查和修正：

```powershell
cd C:\Users\neko\Desktop\github\med-image-seg
uv run med-image-seg --image "D:\images\scan.nii.gz"
```

没有网络时可要求只使用已经校验的缓存：

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz --offline
```

也可以指定本地模型、缓存目录或 ONNX Runtime provider：

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz `
  --model D:\models\cbct-bone-intraoperative-v4.onnx `
  --provider CPUExecutionProvider
```

无需权重的旧自适应算法仍可显式调用，并允许改变处理分辨率：

```powershell
uv run cbct-bone scan.nii.gz bone_mask.nii.gz `
  --backend adaptive --spacing 1.0
```

默认模型缓存在 Windows 的 `%LOCALAPPDATA%\cbct-bone\models`，Linux/macOS 使用标准用户缓存目录。设置 `CBCT_BONE_CACHE_DIR` 或传入 `--cache-dir` 可以覆盖该位置。发布模型的大小和 SHA-256 固定在代码注册表中，校验失败的文件不会进入有效缓存。

模型下载每次请求都会读取标准代理环境变量 `HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY`，同时兼容对应的小写变量。例如：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:10808"
$env:HTTPS_PROXY = "http://127.0.0.1:10808"
uv run cbct-bone scan.nii.gz bone_mask.nii.gz
```

代理配置由 Python 标准库按当前操作系统规则解析；代码不会把代理地址、账号或密码写入模型缓存。

查看所有参数：

```powershell
uv run cbct-bone --help
```

## 标注工具

多边形标注、自动插值、实时 JSON 保存与 GUI 已迁移到独立的
`med-image-seg` 项目。`cbct-bone` 现在只包含开箱即用的骨骼分割运行时，
不会安装 PySide6、PyTorch 或原型训练依赖。

## Python API

最简单的 API 会自动读取影像、获取 v4 权重、执行推理并写出掩码：

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

推荐使用可组合的类接口：

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

可以提前准备权重，以便后续完全离线运行：

```python
from cbct_bone import ModelManager

model_path = ModelManager().get()
print(model_path)
```

已有 NumPy LPS 体数据时，可以单独复用模型类：

```python
from cbct_bone import V4CBCTBoneSegmenter

mask, diagnostics = V4CBCTBoneSegmenter().segment(
    volume,
    spacing_mm=(0.8, 0.8, 0.8),
)
```

`CBCTBoneExtractor` 采用依赖注入。自定义实现只需提供与 `VolumeReader`、`BoneMaskSegmenter` 或 `MaskWriter` 协议一致的方法，便可替换某个阶段。

## 算法选择

默认 v4 是面向当前目标术中 CBCT 设备的 2.5D U-Net。它逐层使用当前层及上下相邻层，在每个体数据内做稳健灰度归一化，并使用训练预标注阶段相同的阈值与二维连通域后处理。当前训练集只有四个病例、208 张人工切片，因此它应被视为减少标注工作量的设备原型，不应据此宣称跨设备或全身泛化能力。完整限制见 [MODEL_CARD.md](MODEL_CARD.md)。

`adaptive` 后端保留了无解剖先验的 3D 自适应方法：

1. 通过三类 multi-Otsu 直方图自动估计弱骨与强骨阈值。
2. 在毫米尺度的 3D 邻域中计算局部均值和方差，抑制仅在全局灰度上类似骨的软组织和伪影。
3. 从强骨种子出发，在弱骨候选内进行 3D 滞后区域生长，恢复部分容积效应造成的低密度边界。
4. 使用物理尺度闭合、封闭腔填充和最小连通域过滤得到最终二值掩码。

这个后备设计来自 Zhang 等人的“初始分类 + 3D 相关迭代 + 区域生长”自适应骨分割思路，并结合现代 multi-Otsu 与三维形态学实现。它无需训练数据和权重，但不具备深度模型的解剖语义。

如果设备灰度分布特殊，可以显式覆盖自动阈值：

```powershell
uv run cbct-bone scan.nii.gz mask.nii.gz --backend adaptive `
  --weak-threshold 180 --strong-threshold 450
```

弱阈值必须小于强阈值。建议先查看 `--json` 输出的自动阈值，再基于少量已审核病例确定设备级参数。

## 读取与坐标约定

- 读取路径只调用 `ct_mri_dicom_nii_reader` 的公开加载函数；本项目没有第二套医疗影像读取实现。
- 输入在处理前重采样为各向同性 `(L, P, S)` 数组。
- NIfTI 写出使用 SimpleITK，但仅用于写入，不用于读取。
- 输出 NIfTI 保留重采样后网格的 spacing 和 LPS origin，direction 为单位矩阵。
- `.ubd.npz` 格式不保存物理 origin，适合包内快速交换，不适合依赖患者坐标的跨软件工作流。

## 验证状态

v4 使用四例目标设备 CBCT 中的 208 张人工切片训练；其中 165 张用于训练、43 张空间留出切片用于内部验证。内部 Dice 为 `0.8678`、Precision 为 `0.8493`、Recall 为 `0.8872`。相邻切片来自相同病例，因此这些数字不是独立患者测试，也不能估计真实临床泛化性能。

自动测试覆盖模型缓存、断点续传、SHA-256 失败、离线模式、ONNX 轴向还原、间距约束、NIfTI 几何往返、CLI 和面向对象依赖注入。ONNX 导出与原始 PyTorch v4 的数值对照最大绝对误差为 `1.58e-5`。合成测试用于防止算法和几何回归，不等同于临床验证。

在真实设备上投入使用前，至少应按设备、部位、剂量协议和金属植入物情况建立带人工金标准的测试集，并报告 Dice、表面 Dice、HD95、灵敏度和假阳性体积。

开源 SynthRAD2023 骨盆 CBCT 的多中心测试已证明纯阈值方案存在明显的设备域偏移。v4 同样尚未在独立病例、其他设备、膝关节或全身各部位上完成验证。当前自动算法不应视为临床充分验证；后续需要按患者隔离独立测试集，并报告 Dice、表面 Dice、HD95、灵敏度和假阳性体积。

## 模型发布

- Release：`model-v4.0.0`
- 文件：`cbct-bone-intraoperative-v4.onnx`
- 大小：`4,428,600` 字节
- SHA-256：`f18ed7c9071b35c613ff748088bfadbd5bd7691793c1c8b80bf9013315c7abc6`
- 训练网格：`0.8 mm` 各向同性 LPS
- 分割阈值：`0.775`

运行时只会接受与注册大小和 SHA-256 同时匹配的模型文件。

运行质量检查：

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=cbct_bone
uv build
```

## 文献依据

- Zhang J, Yan C-H, Chui C-K, Ong S-H. [Fast segmentation of bone in CT images using 3D adaptive thresholding](https://doi.org/10.1016/j.compbiomed.2009.11.020). Computers in Biology and Medicine. 2010;40(2):231-236.
- van Eijnatten M, et al. [CT image segmentation methods for bone used in medical additive manufacturing](https://doi.org/10.1016/j.medengphy.2017.10.008). Medical Engineering & Physics. 2018;51:6-16.
- Isensee F, et al. [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z). Nature Methods. 2021;18:203-211.
- Dot G, et al. [DentalSegmentator: robust open source deep learning-based CT and CBCT image segmentation](https://doi.org/10.1016/j.jdent.2024.105130). Journal of Dentistry. 2024;147:105130.

## 许可证

[MIT](LICENSE)
