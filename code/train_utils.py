"""Conv-SNN 训练、评估、早停和实验文件保存工具。"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from spikingjelly.activation_based import functional
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(slots=True)
class EvaluationResult:
    """一次完整训练集或测试集遍历后的指标。"""

    loss: float
    accuracy: float
    macro_f1: float
    sample_count: int
    confusion_matrix: np.ndarray

    def scalar_dict(self) -> dict[str, float | int]:
        """只返回适合写入 history.csv 的标量字段。"""

        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "sample_count": self.sample_count,
        }


@dataclass(slots=True)
class EarlyStopping:
    """按测试准确率早停；准确率相同时选择测试损失更低的 epoch。

    按用户要求，官方测试集在这里被当作验证集使用。调用方应在实验记录中明确
    标记这一点，不能把由此得到的最高测试准确率视为无偏最终结果。
    """

    patience: int = 10
    min_delta: float = 0.0
    best_accuracy: float = -np.inf
    best_loss: float = np.inf
    best_epoch: int = -1
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience 必须为正整数。")
        if self.min_delta < 0:
            raise ValueError("min_delta 不能为负数。")

    def step(
        self,
        accuracy: float,
        loss: float,
        epoch: int,
    ) -> tuple[bool, bool]:
        """更新早停状态，返回 ``(是否改进, 是否应停止)``。"""

        accuracy_improved = accuracy > self.best_accuracy + self.min_delta
        accuracy_tied = np.isclose(
            accuracy,
            self.best_accuracy,
            rtol=0.0,
            atol=max(self.min_delta, 1e-12),
        )
        tie_loss_improved = accuracy_tied and loss < self.best_loss
        improved = bool(accuracy_improved or tie_loss_improved)

        if improved:
            self.best_accuracy = float(accuracy)
            self.best_loss = float(loss)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        """返回可序列化状态。"""

        return asdict(self)


def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def configure_cuda_performance(
    device: torch.device,
    deterministic: bool = False,
    allow_tf32: bool = True,
) -> None:
    """为固定输入形状的 CUDA 训练启用高吞吐后端设置。

    ``deterministic=False`` 会让 cuDNN 为卷积自动选择更快的算法。TF32 只影响
    未被混合精度覆盖的 float32 矩阵乘法和卷积；需要逐位复现时应关闭这两个
    优化。
    """

    if device.type != "cuda":
        return

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def recommended_num_workers(max_workers: int = 8) -> int:
    """根据平台返回保守的 DataLoader worker 数。

    Linux 使用 ``fork`` 时可共享只读的 STAG 压力数组；Windows 使用 ``spawn``
    会复制这块大数组，因此仍默认单进程加载。
    """

    if max_workers < 0:
        raise ValueError("max_workers 不能为负数。")
    if os.name == "nt" or max_workers == 0:
        return 0
    cpu_count = os.cpu_count() or 1
    return min(max_workers, max(1, cpu_count - 1))


def seed_worker(worker_id: int) -> None:
    """供 DataLoader 使用，使每个 worker 的 NumPy/Python 随机数可复现。"""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _batch_metrics(
    targets: list[np.ndarray],
    predictions: list[np.ndarray],
    num_classes: int,
) -> tuple[float, float, np.ndarray]:
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    accuracy = float((y_true == y_pred).mean())
    labels = np.arange(num_classes)
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return accuracy, macro_f1, matrix


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    grad_clip: float | None = 1.0,
    amp_dtype: torch.dtype | None = None,
    progress_loss_interval: int = 0,
    show_progress: bool = True,
) -> EvaluationResult:
    """训练一个 epoch，并确保每个批次后重置所有 SNN 状态。

    指标张量会留在设备上，直到 epoch 结束后才一次性复制到 CPU，避免每个
    batch 的 ``item()``/``cpu()`` 强制同步 CUDA。``progress_loss_interval``
    大于 0 时才会按指定批次数刷新一次 loss。
    """

    if progress_loss_interval < 0:
        raise ValueError("progress_loss_interval 不能为负数。")

    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    sample_count = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    iterator = tqdm(
        data_loader,
        desc="训练",
        leave=False,
        disable=not show_progress,
    )
    for batch_index, batch in enumerate(iterator, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        finally:
            # 即使某个批次报错，也尽量清理膜电位，避免之后的调试结果被污染。
            functional.reset_net(model)

        batch_size = y.numel()
        total_loss += loss.detach().float() * batch_size
        sample_count += batch_size
        prediction = logits.detach().argmax(dim=1)
        all_targets.append(y.detach())
        all_predictions.append(prediction)
        if (
            progress_loss_interval > 0
            and batch_index % progress_loss_interval == 0
        ):
            iterator.set_postfix(
                loss=f"{loss.detach().float().item():.4f}"
            )

    targets = torch.cat(all_targets).cpu().numpy()
    predictions = torch.cat(all_predictions).cpu().numpy()
    accuracy, macro_f1, matrix = _batch_metrics(
        [targets], [predictions], num_classes
    )
    return EvaluationResult(
        loss=float(total_loss.item()) / sample_count,
        accuracy=accuracy,
        macro_f1=macro_f1,
        sample_count=sample_count,
        confusion_matrix=matrix,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    amp_dtype: torch.dtype | None = None,
    show_progress: bool = True,
) -> EvaluationResult:
    """在验证集或测试集上评估，并在每个批次后重置 SNN 状态。"""

    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    sample_count = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    iterator = tqdm(
        data_loader,
        desc="评估",
        leave=False,
        disable=not show_progress,
    )
    for batch in iterator:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        try:
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                logits = model(x)
                loss = criterion(logits, y)
        finally:
            functional.reset_net(model)

        batch_size = y.numel()
        total_loss += loss.float() * batch_size
        sample_count += batch_size
        prediction = logits.argmax(dim=1)
        all_targets.append(y)
        all_predictions.append(prediction)

    targets = torch.cat(all_targets).cpu().numpy()
    predictions = torch.cat(all_predictions).cpu().numpy()
    accuracy, macro_f1, matrix = _batch_metrics(
        [targets], [predictions], num_classes
    )
    return EvaluationResult(
        loss=float(total_loss.item()) / sample_count,
        accuracy=accuracy,
        macro_f1=macro_f1,
        sample_count=sample_count,
        confusion_matrix=matrix,
    )


def build_history_row(
    epoch: int,
    learning_rate: float,
    train_result: EvaluationResult,
    test_result: EvaluationResult,
) -> dict[str, float | int]:
    """把一轮训练/测试指标压平为一行。"""

    return {
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
        "train_loss": train_result.loss,
        "train_accuracy": train_result.accuracy,
        "train_macro_f1": train_result.macro_f1,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "test_macro_f1": test_result.macro_f1,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    class_names: Iterable[str],
    model_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    early_stopping: EarlyStopping,
    experiment_config: Mapping[str, Any],
) -> None:
    """保存可恢复训练的完整 checkpoint。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "class_names": list(class_names),
        "model_config": dict(model_config),
        "metrics": dict(metrics),
        "early_stopping": early_stopping.state_dict(),
        "experiment_config": dict(experiment_config),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """加载模型，并可选恢复优化器和学习率调度器。"""

    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def _json_safe(value: Any) -> Any:
    """递归转换 NumPy、Path 等 JSON 不认识的对象。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """以 UTF-8 JSON 保存配置或实验摘要。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_history_csv(
    path: str | Path,
    history: list[Mapping[str, Any]],
) -> None:
    """保存每个 epoch 的标量训练记录。"""

    if not history:
        raise ValueError("history 不能为空。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def prepare_output_directory(
    root: str | Path,
    experiment_name: str,
) -> Path:
    """创建并返回本次实验输出目录。"""

    if not experiment_name.strip():
        raise ValueError("experiment_name 不能为空。")
    # 避免实验名意外形成多级或绝对路径。
    safe_name = experiment_name.replace("/", "_").replace("\\", "_")
    output_dir = Path(root) / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def recommended_device() -> torch.device:
    """优先使用 CUDA；没有可用显卡时回退到 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_windows_dataloader_defaults() -> None:
    """为 Windows 多进程训练提供保守默认值的说明性入口。

    当前 Dataset 持有完整压力数组。Windows 的 spawn worker 可能复制这块内存，
    因此 notebook 默认 ``num_workers=0``。这个函数不修改系统设置，仅用于让该
    约束在公开训练工具接口中清晰可见。
    """

    if os.name != "nt":
        return
