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
- RTX PRO 6000 上默认 batch size 512
- 验证默认使用 batch size 1024
- Adam，learning rate `1e-3`
- 训练压力图高斯噪声标准差 `0.015`
- 16 个泊松 SNN 仿真时间步

## Linux RTX PRO 6000 加速

完整模式会针对 Linux 高性能 GPU 自动启用：

- 最多 8 个持久化 DataLoader worker、固定内存和 4 倍预取；
- 一次性矢量化归一化缓存，训练期间不再逐样本处理压力图；
- BF16 自动混合精度、TF32、cuDNN autotune 和 fused Adam；
- GPU 上累计整轮指标，仅在 epoch 末传回 CPU；
- 训练与验证分别显示 `tqdm` 进度条，每隔若干批次刷新一次 loss。

RTX PRO 6000 默认从 batch size 512 开始。若显存不足，可在 Notebook 配置
单元中改小，或在启动前设置环境变量：

```bash
export SNN_BATCH_SIZE=256
export SNN_VALIDATION_BATCH_SIZE=512
export SNN_NUM_WORKERS=8
```

若显存仍有大量余量，可以尝试 `SNN_BATCH_SIZE=768` 或 `1024`。应以每轮耗时
和 `nvidia-smi` 的持续利用率判断吞吐提升，而不是只观察瞬时峰值。

结果写入 `outputs/single_frame_full/`，包括：

- `best_model.pt`、`last_model.pt`
- `history.csv`、`summary.json`、`class_mapping.json`
- `training_curves.png`、`confusion_matrix.png`、`class_accuracy.png`

这是一套遵循论文单帧数据协议的轻量 SNN 基线，并非论文 ResNet 架构或论文
27 类准确率的严格复现。
