# STAG 连续时序 Conv-SNN

本目录提供一套以 recording 为单位构建 STAG 连续触觉窗口，并使用
PyTorch + SpikingJelly 训练卷积脉冲神经网络的清晰基线。

代码不会解压或改写原始数据。它直接读取：

```text
../stag_data/classification_lite.zip
```

## 重要实验说明

按照当前实验要求，官方测试集会在每个 epoch 后参与评估，并用于早停和选择
`best_model.pt`。因此它实际承担了验证集的角色，最佳测试准确率存在模型选择
偏倚，不能再视为严格独立、无偏的最终测试结果。`train_stag_snn.ipynb`、
checkpoint 和 `summary.json` 都会保留这一说明。

## 文件职责

- `stag_data.py`：读取 ZIP/MAT、推导 548 点掩码、检测连续片段、生成窗口清单。
- `stag_dataset.py`：按清单动态读取窗口并归一化为 PyTorch Tensor。
- `snn_model.py`：多步 Conv-SNN 模型。
- `train_utils.py`：训练、评估、早停、指标和实验文件保存。
- `visualization.py`：数据分布、触觉帧、训练曲线和混淆矩阵。
- `stag_explore.ipynb`：只探索数据，不包含训练逻辑。
- `train_stag_snn.ipynb`：参数、训练过程、测试早停和结果可视化。
- `smoke_test.py`：不进行正式训练的真实数据小批次检查。

## 安装依赖

建议在独立 Python 3.11 环境中安装。若要使用 NVIDIA GPU，优先按
[PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择与 CUDA
匹配的安装命令，然后安装其余依赖：

```powershell
python -m pip install -r requirements.txt
```

代码固定使用 PyPI 稳定版 `spikingjelly==0.0.0.0.14` 的
`spikingjelly.activation_based` 接口，不需要 CuPy。

## 推荐运行顺序

1. 打开并逐节运行 `stag_explore.ipynb`，理解字段、recording、掩码和断点。
2. 执行安全冒烟测试：

   ```powershell
   python smoke_test.py
   ```

3. 打开 `train_stag_snn.ipynb`。默认 `RUN_TRAINING = False`，整本 notebook
   只会加载数据并执行一个真实批次前向检查。
4. 确认路径和显存后，将 `RUN_TRAINING` 改为 `True`，再运行训练单元。

## 默认数据定义

- 过滤 `empty_hand`，分类标签为 26 个真实物体。
- 官方 `splitId=0` 为训练，`splitId=1` 为测试。
- 窗口长度 16、步长 8。
- `interaction` 模式要求窗口的 `hasValidLabel` 比例不低于 0.5。
- 归一化为 `clip((x-500)/(650-500), 0, 1)`。
- 大于 950 的异常压力帧会成为断点。
- 帧号跳跃、时间戳不递增或时间间隔过大也会成为断点。

当前数据包在默认配置下应生成：

| 划分 | recording | 窗口 |
|---|---:|---:|
| train | 52 | 8,315 |
| test | 26 | 3,352 |

切换到 `mode="stable"` 后，只保留全部 16 帧均有效的窗口。

## 输出文件

正式训练会写入 `outputs/<EXPERIMENT_NAME>/`：

- `config.json`：完整参数和测试集参与早停的说明；
- `class_mapping.json`：模型标签与原始 `objectId`；
- `train_windows.csv`、`test_windows.csv`：窗口索引清单；
- `history.csv`：每轮训练/测试的 loss、accuracy、macro-F1；
- `best_model.pt`：测试准确率最高、同分时测试损失最低的 checkpoint；
- `last_model.pt`：停止训练时的 checkpoint；
- `summary.json`：最佳 epoch、早停状态和最终指标；
- `training_curves.png`、`confusion_matrix.png`：结果图。

窗口 CSV 不含压力矩阵，因此不会重复保存大量相邻帧。

## 显存与 DataLoader

当前机器的 NVIDIA MX450 只有 2 GB 显存，默认 batch size 为 16。若发生显存
不足，只需在训练 notebook 中把它降到 8 或 4。Windows 下默认
`num_workers=0`，避免 DataLoader 子进程复制完整压力数组。

