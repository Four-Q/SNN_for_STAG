"""STAG 数据、训练曲线和分类结果的可视化函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stag_data import STAGMetadata, build_recording_indices


def _to_numpy(value: object) -> np.ndarray:
    """兼容 NumPy 数组和 CPU/GPU PyTorch Tensor。"""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _save_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")


def plot_sensor_mask(
    sensor_mask: np.ndarray,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """显示 32×32 空间布局中的 548 个真实传感器位置。"""

    mask = _to_numpy(sensor_mask).squeeze().astype(bool)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(mask, cmap="Greens", vmin=0, vmax=1)
    ax.set_title(f"STAG 有效传感器掩码（{int(mask.sum())} 个）")
    ax.set_xlabel("列")
    ax.set_ylabel("行")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_tactile_frame(
    frame: np.ndarray,
    sensor_mask: np.ndarray | None = None,
    title: str = "STAG 触觉压力帧",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """绘制单个 32×32 触觉压力热力图。"""

    pressure = _to_numpy(frame).squeeze().astype(np.float32)
    if sensor_mask is not None:
        mask = _to_numpy(sensor_mask).squeeze().astype(bool)
        pressure = np.ma.masked_where(~mask, pressure)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(pressure, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("列")
    ax.set_ylabel("行")
    fig.colorbar(image, ax=ax, label="压力值")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_tactile_sequence(
    sequence: np.ndarray,
    timestamps: Sequence[float] | None = None,
    sensor_mask: np.ndarray | None = None,
    max_frames: int = 8,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """等间隔选择若干时间步，展示压力随时间的空间变化。"""

    values = _to_numpy(sequence)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError("sequence 应为 [T, 32, 32] 或 [T, 1, 32, 32]。")

    count = min(max_frames, len(values))
    selected = np.linspace(0, len(values) - 1, count, dtype=int)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.5 * rows),
        squeeze=False,
    )
    global_min = float(values.min())
    global_max = float(values.max())
    mask = (
        _to_numpy(sensor_mask).squeeze().astype(bool)
        if sensor_mask is not None
        else None
    )

    for ax, frame_index in zip(axes.ravel(), selected):
        frame = values[frame_index]
        if mask is not None:
            frame = np.ma.masked_where(~mask, frame)
        ax.imshow(frame, cmap="magma", vmin=global_min, vmax=global_max)
        if timestamps is None:
            ax.set_title(f"t={frame_index}")
        else:
            ax.set_title(f"t={frame_index}, {timestamps[frame_index]:.2f}s")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.ravel()[count:]:
        ax.axis("off")
    fig.suptitle("连续触觉压力序列", y=1.01)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, axes


def plot_class_distribution(
    manifest: pd.DataFrame,
    class_names: Sequence[str],
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """比较不同官方划分中的窗口类别数量。"""

    counts = (
        manifest.groupby(["split", "label"])
        .size()
        .rename("windows")
        .reset_index()
    )
    counts["object_name"] = counts["label"].map(dict(enumerate(class_names)))

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=counts,
        x="object_name",
        y="windows",
        hue="split",
        ax=ax,
    )
    ax.set_title("各类别时序窗口数量")
    ax.set_xlabel("物体类别")
    ax.set_ylabel("窗口数")
    ax.tick_params(axis="x", rotation=65)
    ax.legend(title="划分")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_recording_timeline(
    metadata: STAGMetadata,
    batch_id: int,
    recording_id: int,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """显示一条 recording 的最大压力、平均压力和有效标记。"""

    key = (int(batch_id), int(recording_id))
    index = build_recording_indices(metadata)
    if key not in index:
        raise KeyError(f"不存在 recording {key}。")
    indices = index[key]
    time = metadata.timestamp[indices]
    valid_values = metadata.pressure[indices][:, metadata.sensor_mask]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(time, valid_values.mean(axis=1), label="有效传感器平均压力")
    ax.plot(time, valid_values.max(axis=1), label="有效传感器最大压力", alpha=0.8)
    valid = metadata.has_valid_label[indices]
    ax.fill_between(
        time,
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        where=valid,
        color="green",
        alpha=0.12,
        label="hasValidLabel=True",
    )
    ax.set_title(
        f"recording 时间线：{metadata.recordings[recording_id]}"
    )
    ax.set_xlabel("相对时间（秒）")
    ax.set_ylabel("原始压力")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_training_curves(
    history: Sequence[Mapping[str, float]],
    best_epoch: int | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """绘制训练/测试损失、准确率和 macro-F1。"""

    history_df = pd.DataFrame(history)
    required = {
        "epoch",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "train_macro_f1",
        "test_macro_f1",
    }
    missing = required.difference(history_df.columns)
    if missing:
        raise ValueError(f"history 缺少列：{sorted(missing)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = (
        ("loss", "损失"),
        ("accuracy", "准确率"),
        ("macro_f1", "Macro-F1"),
    )
    for ax, (column, title) in zip(axes, pairs):
        ax.plot(
            history_df["epoch"],
            history_df[f"train_{column}"],
            label="训练",
        )
        ax.plot(
            history_df["epoch"],
            history_df[f"test_{column}"],
            label="测试（用于早停）",
        )
        if best_epoch is not None:
            ax.axvline(
                best_epoch,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="最佳 epoch",
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, axes


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    normalize: bool = True,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """绘制 26 类混淆矩阵；默认按真实类别归一化。"""

    values = _to_numpy(matrix).astype(np.float64)
    annotation_format = "d"
    if normalize:
        row_sum = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_sum,
            out=np.zeros_like(values),
            where=row_sum != 0,
        )
        annotation_format = ".2f"
    else:
        values = values.astype(np.int64)

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        values,
        cmap="Blues",
        annot=True,
        fmt=annotation_format,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "比例" if normalize else "样本数"},
    )
    ax.set_title("测试集混淆矩阵（测试集同时用于早停）")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.tick_params(axis="x", rotation=65)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_prediction_examples(
    x: object,
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    sensor_mask: np.ndarray,
    max_examples: int = 6,
) -> tuple[plt.Figure, np.ndarray]:
    """使用每个窗口中间帧展示若干预测样本。"""

    windows = _to_numpy(x)
    if windows.ndim != 5:
        raise ValueError("x 应为 [B, T, C, H, W]。")
    count = min(max_examples, len(windows))
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        squeeze=False,
    )
    mask = _to_numpy(sensor_mask).squeeze().astype(bool)
    for index, ax in enumerate(axes.ravel()[:count]):
        middle_frame = windows[index, windows.shape[1] // 2, 0]
        ax.imshow(np.ma.masked_where(~mask, middle_frame), cmap="magma")
        true_name = class_names[int(targets[index])]
        pred_name = class_names[int(predictions[index])]
        color = "green" if true_name == pred_name else "red"
        ax.set_title(f"真实：{true_name}\n预测：{pred_name}", color=color)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[count:]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes

