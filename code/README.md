# STAG 单帧压力图 Conv-SNN

本目录提供一套易于顺序阅读的单帧物体识别基线。每个样本只包含一张
`1×32×32` 压力图；模型内部生成的 16 个泊松时间步是 SNN 仿真时间，
不是 16 张连续触觉帧。

## 数据口径

- 直接读取 `../stag_data/classification_lite.zip` 中的 `metadata.mat`，不解压。
- 使用官方 `splitId` 和 `isBalanced` 标记，不随机打散相邻帧。
- 按当前实验要求移除 `empty_hand`，识别 26 个真实物体。
- 完整训练集为 35,178 帧（每类 1,353），验证集为 15,522 帧（每类 597）。
- 压力归一化为 `clip((x-500)/(650-500), 0, 1)`，并清零无效传感器位置。

需要特别注意：官方 `test` 在 notebook 中按当前要求被用作每个 epoch 的
validation，并参与 `best_model.pt` 选择。因此最佳验证准确率存在模型选择偏差，
不能被描述为无偏的最终测试结果。

## 文件职责

- `data.py`：ZIP/MAT 读取、官方单帧划分、传感器掩码与 Dataset。
- `model.py`：泊松编码和三层轻量 Conv-SNN。
- `train_single_frame_snn.ipynb`：训练、验证、可视化和模型保存的唯一入口。
- `smoke_test.py`：无界面执行 notebook 的 smoke 模式并检查全部产物。

## 安装

建议使用 Python 3.11。若使用 NVIDIA GPU，应先按 PyTorch 官方说明安装与
CUDA 匹配的 PyTorch，再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
```

## 冒烟测试

```powershell
python smoke_test.py
```

该测试读取真实 STAG 数据，以 CPU、少量样本、两个 SNN 时间步运行一个
训练/验证 epoch，并在临时目录检查 checkpoint、CSV、JSON 和三张英文图表。
它不会修改已提交 notebook，也不会进行完整训练。

## 完整训练

打开 `train_single_frame_snn.ipynb`，将配置单元中的：

```python
RUN_MODE = "full"
```

然后按顺序运行全部单元。完整模式默认参数为：

- 200 epochs
- batch size 32
- Adam，learning rate `1e-3`
- 训练压力图高斯噪声标准差 `0.015`
- 16 个泊松 SNN 仿真时间步

结果写入 `outputs/single_frame_full/`，包括：

- `best_model.pt`、`last_model.pt`
- `history.csv`、`summary.json`、`class_mapping.json`
- `training_curves.png`、`confusion_matrix.png`、`class_accuracy.png`

这是一套遵循论文单帧数据协议的轻量 SNN 基线，并非论文 ResNet 架构或论文
27 类准确率的严格复现。
