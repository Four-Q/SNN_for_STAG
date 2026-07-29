# STAG 单帧物体识别 Conv-SNN 工作日志

## 1. 日志信息

| 项目 | 内容 |
|---|---|
| 记录日期 | 2026-07-28 |
| Git 仓库 | `Four-Q/SNN_for_STAG` |
| 开发分支 | `single-frame-classify` |
| 基础分支 | `main` |
| 分支起点 | `9beeb75` |
| 日志编写时最新提交 | `62dba06 Optimize RTX PRO 6000 training` |
| 远程分支 | `origin/single-frame-classify` |
| 主要运行平台 | Linux、NVIDIA RTX PRO 6000 |
| 本地验证平台 | Windows、Python 3.11、PyTorch 2.7.1 CPU |

本日志记录从原有 STAG 连续时序分类代码转向单帧压力图物体识别基线的完整过程，
包括需求决策、数据口径、模型结构、Notebook 工作流、冒烟测试、Git 发布以及
RTX PRO 6000 训练加速。

---

## 2. 背景与问题

原有代码将同一条 recording 中的连续压力帧构造成时序窗口，并用 Conv-SNN
识别物体。实际实验中发现，这一时序识别方案的测试集准确率很低，因此决定先
回到 Nature 论文《Learning the signatures of the human grasp using a
scalable tactile glove》的单帧物体识别任务，建立一个更简单、更容易解释和
排查问题的 SNN 基线。

新方案不再把多张真实压力图作为 SNN 的时间维度。**每个样本只包含一张**
**`1×32×32` 压力图；SNN 所需的时间维度在模型内部通过泊松频率编码生成。**

该调整的主要目标是：

1. 消除窗口构造、阶段识别和连续帧冗余带来的复杂度；
2. **严格复用官方单帧训练/测试划分，避免相邻帧随机划分造成数据泄漏；**
3. 保留 SNN 模型，验证单帧压力分布是否能够提供可学习的物体特征；
4. 将训练、验证、结果可视化和模型保存集中在一个 Notebook 中；
5. 为 Linux RTX PRO 6000 远程训练提供高吞吐配置。

---

## 3. 最终确定的实验口径

### 3.1 数据集

代码直接读取：

```text
stag_data/classification_lite.zip
```

ZIP 文件中的主要数据来自 `metadata.mat`。代码不会把 ZIP 解压到工作目录，
也不会改写原始数据。

已验证的数据事实如下：

| 数据项目 | 数量 |
|---|---:|
| 原始压力帧 | 135,187 |
| 压力图尺寸 | 32×32 |
| 矩阵位置总数 | 1,024 |
| 有效物理传感器 | 548 |
| 无效填充位置 | 476 |
| 官方类别 | 27 |
| 本项目保留类别 | 26 |

**官方 27 类中包含 `empty_hand`。按照当前实验要求，本项目移除**
**`empty_hand`，只识别 26 个真实物体。**

### 3.2 划分与平衡

划分完全使用官方字段：

- `splitId`：区分官方 `train` 和 `test`；
- `isBalanced`：选择官方类别平衡子集；
- `hasValidLabel`：用于验证 `isBalanced` 样本均为有效接触帧。

移除 `empty_hand` 后的数量为：

| 划分 | 每类帧数 | 类别数 | 总帧数 |
|---|---:|---:|---:|
| 训练集 | 1,353 | 26 | 35,178 |
| 验证集（官方 test） | 597 | 26 | 15,522 |

代码会验证训练集和验证集的 `(batchId, recordingId)` 不重叠。这样不会把同一
条连续 recording 中的相邻帧随机分到训练集和验证集。

### 3.3 验证集使用方式

按照当前实验选择，官方 `test` 会在每个 epoch 中作为 validation 使用，并参与
`best_model.pt` 的选择。

这一做法存在明确限制：

> 官方测试集参与了逐 epoch 验证和最佳模型选择，因此得到的最佳验证准确率存在
> 模型选择偏差，不能被表述为无偏的最终测试准确率。

这一警告已经写入：

- Notebook 的 Markdown；
- Notebook 的运行输出；
- `best_model.pt` 和 `last_model.pt`；
- `summary.json`；
- [`code/README.md`](../../code/README.md)。

---

## 4. 分支与代码重构

### 4.1 分支

从干净的 `main` 创建：

```text
single-frame-classify
```

远程地址：

```text
git@github.com:Four-Q/SNN_for_STAG.git
```

### 4.2 删除的旧实现

原有连续时序代码被移除，包括：

- `code/stag_data.py`
- `code/stag_dataset.py`
- `code/stag_explore.ipynb`
- `code/train_stag_snn.ipynb`
- `code/train_utils.py`
- `code/visualization.py`

这些文件包含 recording 窗口构造、连续片段检测、多帧 Dataset、旧训练工具、
时序探索以及旧结果可视化，不再属于单帧任务。

### 4.3 最终代码结构

| 文件 | 职责 |
|---|---|
| [`code/data.py`](../../code/data.py) | 读取 ZIP/MAT、恢复传感器掩码、构造官方单帧划分和 Dataset |
| [`code/model.py`](../../code/model.py) | 泊松编码器和轻量 Conv-SNN |
| [`code/train_single_frame_snn.ipynb`](../../code/train_single_frame_snn.ipynb) | 训练、验证、进度条、可视化、模型保存的唯一入口 |
| [`code/smoke_test.py`](../../code/smoke_test.py) | 无界面执行 Notebook 冒烟模式并验证所有产物 |
| [`code/requirements.txt`](../../code/requirements.txt) | Python 依赖版本 |
| [`code/README.md`](../../code/README.md) | 数据口径、运行方式和 GPU 调优说明 |
| `code/outputs/.gitkeep` | 保留被 Git 忽略的结果目录 |

---

## 5. 数据处理实现

### 5.1 读取方式

`data.py` 使用 `zipfile` 和内存缓冲区读取 `metadata.mat`，再由
`scipy.io.loadmat` 解析需要的字段。

读取后会检查：

1. `pressure` 是否为 `[N, 32, 32]`；
2. 压力值是否为整数；
3. 每个逐帧字段长度是否与 `pressure` 一致；
4. `objectId` 和 `splitId` 是否在合法范围内；
5. `isBalanced=True` 是否必然对应 `hasValidLabel=True`。

### 5.2 传感器掩码

lite 数据中 476 个填充位置始终保持常数，548 个真实传感器位置会随时间变化。
因此掩码使用以下条件恢复：

```python
sensor_mask = pressure.max(axis=0) != pressure.min(axis=0)
```

恢复后的有效位置数量必须严格等于 548，否则代码会直接报错，不会静默继续。

### 5.3 压力归一化

每张压力图使用：

```text
clip((x - 500) / (650 - 500), 0, 1)
```

随后乘以传感器掩码，将 476 个无效位置重新清零。

训练阶段还会对归一化图像加入标准差为 `0.015` 的高斯噪声，并在噪声后再次
裁剪至 `[0,1]`、重新应用传感器掩码。

### 5.4 标签映射

原始 `objectId=0` 对应 `empty_hand`，被排除。其余原始 ID 被连续映射到
模型标签 `0–25`。

每次运行都会保存 `class_mapping.json`，其中包含：

- 模型标签；
- 原始 `objectId`；
- 官方英文类别名。

### 5.5 归一化缓存

最初的实现会在每次 `__getitem__` 中把单帧从整数转换为 `float32` 并归一化。
这会在每个 epoch 中重复执行大量细粒度 CPU 操作，容易导致 GPU 等待数据。

优化后，训练集和验证集会在 Dataset 初始化时一次性矢量化归一化，并保存连续
`float32` 缓存。之后的 `__getitem__` 只创建 Tensor 视图。

缓存实测大小：

```text
198.046875 MiB
```

该缓存包括 35,178 张训练图和 15,522 张验证图。在目标主机 120 GB 内存条件下
占用很小。

---

## 6. 单帧 Conv-SNN 模型

### 6.1 输入与输出

公开输入：

```text
[B, 1, 32, 32]
```

公开输出：

```text
[B, 26]
```

模型只接收单张压力图，不接收真实触觉序列。

### 6.2 泊松频率编码

**归一化压力值被解释为每个内部时间步产生脉冲的概率**。完整训练默认生成
16 个内部时间步：

```text
[B, 1, 32, 32]
    ↓ PoissonEncoder
[T, B, 1, 32, 32]，T=16
```

这些时间步是 SNN 仿真时间，不是 STAG 在 7.3 帧/秒下采集的连续压力帧。

### 6.3 网络结构

轻量模型包含三组卷积脉冲特征层：

```text
Conv2d → BatchNorm → LIF → MaxPool
Conv2d → BatchNorm → LIF → MaxPool
Dropout2d
Conv2d → BatchNorm → LIF → MaxPool
AdaptiveAvgPool → Linear
```

主要参数：

| 参数 | 默认值 |
|---|---:|
| 通道数 | 16 / 32 / 64 |
| LIF `tau` | 2.0 |
| Dropout | 0.2 |
| 内部时间步 | 16 |
| 类别数 | 26 |
| **可训练参数** | **25,098** |

分类头保留为模拟线性读出，对所有内部时间步的 logits 求平均。每个训练或验证
批次完成后调用 SpikingJelly 的 `functional.reset_net`，清除 LIF 膜电位状态。

---

## 7. Notebook 工作流

[`train_single_frame_snn.ipynb`](../../code/train_single_frame_snn.ipynb)
是训练和验证的唯一入口，按以下顺序组织：

1. 导入依赖、设置运行模式、随机种子和设备；
2. 读取官方数据和显示划分统计；
3. 显示单帧压力图示例；
4. 构造训练和验证 DataLoader；
5. 创建 Conv-SNN、优化器和损失函数；
6. 在 Notebook 内定义训练和验证循环；
7. 逐 epoch 训练并使用官方 test 做验证；
8. 保存 best/last checkpoint；
9. 重新加载最佳模型并计算验证结果；
10. 绘制训练曲线、混淆矩阵和各类别准确率；
11. 保存 CSV、JSON、模型和图片。

### 7.1 运行模式

默认值：

```python
RUN_MODE = "smoke"
```

冒烟模式：

- CPU；
- 4 个训练样本；
- 4 个验证样本；
- batch size 4；
- 2 个 SNN 时间步；
- 1 个 epoch。

完整训练需要修改为：

```python
RUN_MODE = "full"
```

### 7.2 完整训练默认参数

| 参数 | 默认值 |
|---|---:|
| Epochs | 200 |
| 训练 batch size | 512（GPU 显存不少于 40 GiB） |
| 验证 batch size | 1024 |
| 学习率 | 1e-3 |
| 优化器 | Adam |
| 损失 | CrossEntropyLoss |
| SNN 时间步 | 16 |
| 高斯噪声标准差 | 0.015 |
| 随机种子 | 42 |

如果 GPU 显存小于 40 GiB，完整模式默认训练 batch size 为 256。没有 CUDA 时
默认 batch size 为 64。

### 7.3 训练与验证进度条

训练、逐 epoch 验证以及最佳模型复核都使用 `tqdm.auto.tqdm`：

- Notebook 中显示训练进度条；
- Notebook 中显示验证进度条；
- 进度单位为“批次”；
- 每隔 10 个批次或最后一个批次刷新 loss；
- 设置 `mininterval=0.5`，降低频繁刷新造成的开销。

依赖中增加：

```text
tqdm>=4.66,<5
ipywidgets>=8,<9
```

其中 `ipywidgets` 用于确保 JupyterLab 能正常显示交互式进度组件。

---

## 8. 结果文件

完整模式默认输出到：

```text
code/outputs/single_frame_full/
```

冒烟模式默认输出到：

```text
code/outputs/single_frame_smoke/
```

生成文件：

| 文件 | 内容 |
|---|---|
| `best_model.pt` | 验证 Top-1 最优、同分时验证 loss 更低的模型 |
| `last_model.pt` | 最后一个 epoch 的模型 |
| `history.csv` | 每轮训练/验证指标和吞吐 |
| `summary.json` | 数据、模型、性能配置、最佳指标和验证偏差说明 |
| `class_mapping.json` | 标签、原始 objectId 和物体名称 |
| `training_curves.png` | loss、Top-1、Top-3、macro-F1 曲线 |
| `confusion_matrix.png` | 验证集混淆矩阵 |
| `class_accuracy.png` | 每类别验证准确率 |

Checkpoint 中保存：

- epoch；
- 模型参数；
- 优化器参数；
- 模型配置；
- 性能配置；
- 类别名称；
- 验证指标；
- 官方 test 被用作 validation 的说明。

### 8.1 指标

每轮记录：

- Cross-entropy loss；
- Top-1 accuracy；
- Top-3 accuracy；
- Macro-F1；
- 样本数；
- 运行秒数；
- samples/s。

指标张量在 GPU 上累计，整轮结束后才统一传回 CPU。这样避免原实现中每个
batch 调用 `.item()`、`int()` 或 `.cpu()` 造成频繁 CUDA 同步。

---

## 9. 图表与语言规范

代码和文档遵循以下约定：

- Python 注释和 docstring 使用中文；
- Notebook Markdown 使用中文；
- Notebook 普通运行提示使用中文；
- README 使用中文；
- 图表中的标题、坐标轴、图例、色条和类别名称使用英文。

图表使用的主要英文文本包括：

- `Single-Frame Conv-SNN Training History`
- `Validation Confusion Matrix (Official Test Split)`
- `Per-Class Validation Accuracy (Official Test Split)`
- `Predicted object`
- `True object`
- `Object class`
- `Accuracy`

官方类别名称原本就是英文标识，因此在映射和图表中保持不变。

---

## 10. RTX PRO 6000 加速

### 10.1 问题现象

远程主机环境：

| 资源 | 配置 |
|---|---|
| 操作系统 | Linux |
| GPU | NVIDIA RTX PRO 6000 |
| GPU 数量 | 1 |
| CPU | 25 vCPU |
| 系统内存 | 120 GB |

监控截图显示初始 GPU 利用率大部分时间约为 10%，偶尔短暂升至约 23%。这说明
小模型并没有持续向 GPU 提交足够大的计算任务，且可能受到数据供给和 CPU/GPU
同步影响。

### 10.2 已实施的优化

#### 增大批次

- 训练默认 batch size：512；
- 验证默认 batch size：1024；
- 显存不足 40 GiB 时训练默认降为 256。

增大批次能减少每个 epoch 的 Python 循环次数，并使卷积、BatchNorm 和多步
LIF 计算形成更大的 GPU kernel 工作量。

#### Linux DataLoader

完整模式在 Linux 默认启用：

- 最多 8 个 worker；
- `pin_memory=True`；
- `persistent_workers=True`；
- `prefetch_factor=4`；
- CUDA 拷贝使用 `non_blocking=True`。

Windows 默认仍使用 `num_workers=0`，避免 `spawn` 复制大型缓存。

#### CUDA 计算后端

CUDA 模式启用：

- BF16 自动混合精度；不支持 BF16 时回退 FP16；
- FP16 回退时使用 `GradScaler`；
- TF32；
- `torch.backends.cudnn.benchmark=True`；
- `torch.set_float32_matmul_precision("high")`；
- fused Adam。

#### 减少同步

原训练循环每个 batch 都把 loss、预测和正确数同步回 CPU。优化后：

- loss、Top-1 和 Top-3 在 GPU 上累计；
- targets 和 predictions 暂存在 GPU；
- epoch 结束时统一传回 CPU；
- tqdm 的 loss 每 10 个 batch 才读取一次。

#### 吞吐记录

`history.csv` 和控制台输出现在会记录：

```text
训练吞吐=... 样本/秒
验证吞吐=... 样本/秒
```

这比只观察 `nvidia-smi` 的瞬时利用率更适合比较不同 batch size 和 worker 数量。

### 10.3 可调环境变量

完整训练前可以设置：

```bash
export SNN_BATCH_SIZE=512
export SNN_VALIDATION_BATCH_SIZE=1024
export SNN_NUM_WORKERS=8
export SNN_TIME_STEPS=16
```

如果显存仍有大量余量，可以依次试验：

```bash
export SNN_BATCH_SIZE=768
```

或：

```bash
export SNN_BATCH_SIZE=1024
```

若出现 CUDA out-of-memory，应先降回 512 或 256。调优时应同时记录：

1. 训练 samples/s；
2. 验证 samples/s；
3. 每个 epoch 总时间；
4. GPU 显存占用；
5. GPU 持续利用率；
6. 验证准确率是否发生明显变化。

### 10.4 尚未完成的性能验证

本地开发环境没有 CUDA，因此以下内容尚未实测：

- RTX PRO 6000 的真实持续利用率；
- batch size 512/768/1024 的显存峰值；
- BF16 下 SpikingJelly 模型的实际吞吐；
- 多 worker 与单 worker 的真实差异；
- 优化前后的 epoch 时间对比。

当前结论是代码已经具备上述加速路径，并通过 CPU 冒烟测试；不能把预期加速
表述为已在 RTX PRO 6000 上得到的测量结果。

---

## 11. 冒烟测试

### 11.1 测试入口

```bash
cd code
python smoke_test.py
```

测试脚本使用 `nbclient` 无界面执行完整 Notebook 的 smoke 模式，并把所有结果
写入临时目录，不修改仓库中的 Notebook 输出。

### 11.2 验证内容

测试会检查：

1. Notebook JSON 格式有效；
2. Notebook 中不存在已提交的 execution count 和 cell outputs；
3. Notebook 确实导入并使用 `tqdm`；
4. 训练和验证调用都传入 tqdm 描述；
5. 原始帧数为 135,187；
6. 有效传感器为 548；
7. 类别数为 26；
8. 训练集为 35,178；
9. 验证集为 15,522；
10. `empty_hand` 已移除；
11. 单帧形状为 `[1,32,32]`；
12. 压力值位于 `[0,1]`；
13. 无效传感器位置保持为 0；
14. 模型前向输出为 `[B,26]`；
15. 一次反向传播和 Adam 更新成功；
16. SNN 状态能够重置；
17. `best_model.pt` 和 `last_model.pt` 可加载；
18. `history.csv` 指标有限；
19. `summary.json` 包含验证偏差和性能配置；
20. 三张英文图表均能正常读取。

### 11.3 已通过的验证

开发过程中多次执行了真实数据 Notebook 冒烟测试，均成功通过。最后一次 GPU
优化后的端到端冒烟测试输出为：

```text
冒烟测试通过：Notebook 无界面执行、真实 STAG 数据、一次训练/验证、
模型文件、指标文件和英文图表均正常。
```

最后一次测试耗时约 52 秒。该时间包含：

- 从 ZIP 读取并解析完整 metadata；
- 恢复传感器掩码；
- 建立约 198 MiB 归一化缓存；
- 微型训练和验证；
- 保存两个 checkpoint；
- 生成 CSV、JSON 和三张图片；
- 再次加载和检查全部产物。

该耗时不是正式训练吞吐基准。

---

## 12. Git 提交与发布记录

### 12.1 核心提交

| 提交 | 说明 | 主要内容 |
|---|---|---|
| `329e4f4` | `Implement single-frame SNN notebook` | 删除旧时序代码，建立单帧数据、模型、Notebook、冒烟测试和文档 |
| `796ddfa` | `Localize notebook documentation` | Notebook Markdown、注释和普通提示中文化，图表保持英文 |
| `62dba06` | `Optimize RTX PRO 6000 training` | 数据缓存、Linux DataLoader、BF16/TF32、fused Adam、指标 GPU 累计和 tqdm |

三个提交均已推送到：

```text
origin/single-frame-classify
```

未创建 Pull Request。

### 12.2 远程主机拉取

已有仓库切换或更新该分支：

```bash
git fetch origin
git switch single-frame-classify
git pull --ff-only
```

首次克隆：

```bash
git clone --branch single-frame-classify --single-branch \
  git@github.com:Four-Q/SNN_for_STAG.git
```

拉取后可验证：

```bash
git branch --show-current
git log -1 --oneline
```

预期为：

```text
single-frame-classify
62dba06 Optimize RTX PRO 6000 training
```

### 12.3 Notebook 本地改动阻止 pull

远程主机曾因执行或修改 Notebook，导致 `git pull` 报错：

```text
Your local changes to the following files would be overwritten by merge:
    code/train_single_frame_snn.ipynb
```

推荐先暂存本地 Notebook：

```bash
git stash push -m "pull前的本地Notebook" \
  -- code/train_single_frame_snn.ipynb
git pull --ff-only
```

不要在 pull 后立即 `git stash pop`，否则可能重新引入旧 Notebook 或产生
Notebook JSON 冲突。应先确认新版本正常，再决定是否恢复本地内容。

如果确定不要本地 Notebook 改动：

```bash
git restore -- code/train_single_frame_snn.ipynb
git pull --ff-only
```

这一操作会丢弃该文件的未提交修改，因此应谨慎使用。

---

## 13. 远程主机完整运行步骤

### 13.1 数据

数据文件被 `.gitignore` 排除，不会随 Git 下载。远程主机必须单独准备：

```text
SNN_for_STAG/stag_data/classification_lite.zip
```

### 13.2 安装依赖

应优先安装与远程 CUDA 匹配的 PyTorch，然后安装项目依赖：

```bash
cd ~/autodl-tmp/SNN_for_STAG/code
python -m pip install -r requirements.txt
```

安装或升级 `ipywidgets` 后，建议重启 Jupyter kernel。

### 13.3 冒烟测试

```bash
python smoke_test.py
```

### 13.4 完整训练

打开：

```text
code/train_single_frame_snn.ipynb
```

将：

```python
RUN_MODE = "smoke"
```

改为：

```python
RUN_MODE = "full"
```

然后按顺序执行全部单元。

建议首次正式训练先使用：

```bash
export SNN_BATCH_SIZE=512
export SNN_VALIDATION_BATCH_SIZE=1024
export SNN_NUM_WORKERS=8
```

训练期间同时观察：

```bash
watch -n 1 nvidia-smi
```

并记录 Notebook 每轮输出的 samples/s。

---

## 14. 科学解释与限制

### 14.1 不是 Nature 网络的严格复现

本项目复用了论文的数据划分、平衡标记、单帧输入、高斯噪声和部分训练设置，
但模型是轻量 Conv-SNN，不是论文中修改后的 ResNet-18。

因此不能把本项目结果描述为 Nature 论文网络的严格复现。

### 14.2 类别口径不同

论文物体识别包含：

```text
26 个真实物体 + empty_hand = 27 类
```

本项目只保留 26 个真实物体。因此准确率不能直接与论文 27 类结果作完全等价
比较。

### 14.3 单帧准确率本身有限

论文单帧基线的 Top-1 约为 37.97%，Top-3 约为 60.43%。单帧只反映一个局部
接触状态，无法包含同一抓握过程中多张互补压力图的信息。

因此本项目的主要作用是建立可解释的单帧 SNN 基线，而不是预设它一定超过
多帧模型。

### 14.4 SNN 时间步不是传感器时间

模型的 16 个泊松时间步来自同一张压力图。它们不能被解释为 16 个真实采样帧，
也不表示约 2.2 秒的 STAG 物理过程。

### 14.5 单参与者泛化

原始分类数据主要来自单一参与者、单只右手和同一套手套。模型可能学习到：

- 特定参与者的手形；
- 特定抓握策略；
- 手套佩戴位置；
- 物体与手形的联合压力模式。

目前尚不能据此声称模型具备跨参与者、跨手型或跨手套泛化能力。

---

## 15. 当前完成状态

### 已完成

- 建立并发布 `single-frame-classify` 分支；
- 删除旧连续时序训练实现；
- 建立 26 类官方平衡单帧 Dataset；
- 实现 548 点传感器掩码；
- 实现泊松编码轻量 Conv-SNN；
- 将训练和验证集中到单一 Notebook；
- 实现 Top-1、Top-3、macro-F1 和混淆矩阵；
- 实现 best/last checkpoint；
- 实现 CSV、JSON 和英文图表保存；
- 全面中文化代码注释和 Notebook Markdown；
- 实现 Notebook 训练/验证 tqdm 进度条；
- 实现 Linux RTX PRO 6000 数据与 CUDA 加速；
- 实现无界面真实数据冒烟测试；
- 将三个核心提交推送到远程分支。

### 尚未完成

- 尚未执行完整 200 epoch 训练；
- 尚未得到正式训练和验证准确率；
- 尚未在 RTX PRO 6000 上测量优化后的持续 GPU 利用率；
- 尚未比较 batch size 512、768、1024 的真实吞吐；
- 尚未建立不参与模型选择的独立最终测试集；
- 尚未进行多次独立训练并报告均值与标准差；
- 尚未与非脉冲 CNN 基线作公平对比；
- 尚未验证跨参与者或跨手套泛化。

---

## 16. 建议的下一步

1. 在 RTX PRO 6000 上运行 smoke test，确认 CUDA 环境和 Jupyter 进度条正常；
2. 用 batch size 512 运行 2–5 个 epoch，记录 samples/s、显存和利用率；
3. 在显存允许时测试 batch size 768 和 1024；
4. 选择吞吐最高且训练稳定的 batch size；
5. 执行完整训练并保存所有结果文件；
6. 检查训练/验证曲线是否存在明显过拟合；
7. 检查混淆矩阵中最容易混淆的物体类别；
8. 固定最终超参数后进行多次随机种子实验；
9. 后续增加相同数据口径的普通 CNN 基线，用于判断 SNN 本身的收益；
10. 若需要无偏最终指标，应从官方 train 内建立 validation，把官方 test 留到
    最终只评估一次。
