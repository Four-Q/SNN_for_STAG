"""STAG 单帧分类数据读取与 PyTorch Dataset。

本模块只做三件事：
1. 直接从 ``classification_lite.zip`` 读取 ``metadata.mat``；
2. 使用官方 ``splitId`` 和 ``isBalanced`` 标记构造单帧训练/验证集；
3. 将一张 32×32 压力图归一化为 PyTorch Tensor。

训练、验证、可视化和模型保存均位于 ``train_single_frame_snn.ipynb``。
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


REQUIRED_FIELDS = (
    "pressure",
    "frame",
    "batchId",
    "recordingId",
    "objects",
    "objectId",
    "splitId",
    "splits",
    "hasValidLabel",
    "isBalanced",
)


@dataclass(slots=True)
class STAGMetadata:
    """从官方 ``metadata.mat`` 读取的单帧分类字段。"""

    pressure: np.ndarray
    frame: np.ndarray
    batch_id: np.ndarray
    recording_id: np.ndarray
    object_id: np.ndarray
    split_id: np.ndarray
    has_valid_label: np.ndarray
    is_balanced: np.ndarray
    objects: tuple[str, ...]
    splits: tuple[str, ...]
    sensor_mask: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.pressure.shape[0])


@dataclass(slots=True)
class FrameSplits:
    """论文官方平衡划分去除 ``empty_hand`` 后的索引和标签映射。"""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    class_names: tuple[str, ...]
    original_object_ids: tuple[int, ...]
    original_to_label: np.ndarray

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _matlab_strings(value: np.ndarray) -> tuple[str, ...]:
    """将 SciPy 读取的 MATLAB cell/string 数组转为 Python 字符串。"""

    result: list[str] = []
    for item in np.atleast_1d(value).ravel():
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        if isinstance(item, bytes):
            result.append(item.decode("utf-8", errors="replace"))
        else:
            result.append(str(item))
    return tuple(result)


def infer_sensor_mask(
    pressure: np.ndarray,
    expected_sensor_count: int = 548,
) -> np.ndarray:
    """由全数据中的时间变化恢复 548 个真实传感器位置。

    lite 数据中 476 个填充位置始终保持常数，真实传感器位置会随时间变化。
    如果未来数据不再满足该性质，函数会明确失败，避免静默使用错误掩码。
    """

    if pressure.ndim != 3 or pressure.shape[1:] != (32, 32):
        raise ValueError(
            f"pressure 应为 [N, 32, 32]，实际得到 {pressure.shape}。"
        )
    sensor_mask = pressure.max(axis=0) != pressure.min(axis=0)
    actual_count = int(sensor_mask.sum())
    if actual_count != expected_sensor_count:
        raise ValueError(
            "无法可靠恢复传感器掩码："
            f"期望 {expected_sensor_count} 点，实际 {actual_count} 点。"
        )
    return sensor_mask


def load_stag_metadata(zip_path: str | Path) -> STAGMetadata:
    """从 ZIP 内存读取 ``metadata.mat``，不在磁盘解压数据。"""

    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"找不到 STAG 数据包：{zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        if "metadata.mat" not in archive.namelist():
            raise KeyError(f"{zip_path} 中不存在 metadata.mat。")
        mat_buffer = io.BytesIO(archive.read("metadata.mat"))

    raw = loadmat(
        mat_buffer,
        variable_names=list(REQUIRED_FIELDS),
        squeeze_me=True,
    )
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise KeyError(f"metadata.mat 缺少字段：{missing}")

    pressure = np.asarray(raw["pressure"])
    if pressure.ndim != 3 or pressure.shape[1:] != (32, 32):
        raise ValueError(f"压力数据形状错误：{pressure.shape}")
    if not np.issubdtype(pressure.dtype, np.integer):
        raise TypeError(f"原始压力应为整数，实际为 {pressure.dtype}。")

    frame_count = pressure.shape[0]

    def vector(name: str, dtype: Any) -> np.ndarray:
        values = np.asarray(raw[name], dtype=dtype).reshape(-1)
        if len(values) != frame_count:
            raise ValueError(
                f"{name} 长度 {len(values)} 与压力帧数 {frame_count} 不一致。"
            )
        return values

    metadata = STAGMetadata(
        pressure=pressure,
        frame=vector("frame", np.int64),
        batch_id=vector("batchId", np.int64),
        recording_id=vector("recordingId", np.int64),
        object_id=vector("objectId", np.int64),
        split_id=vector("splitId", np.int64),
        has_valid_label=vector("hasValidLabel", np.bool_),
        is_balanced=vector("isBalanced", np.bool_),
        objects=_matlab_strings(raw["objects"]),
        splits=_matlab_strings(raw["splits"]),
        sensor_mask=infer_sensor_mask(pressure),
    )
    _validate_metadata(metadata)
    return metadata


def _validate_metadata(metadata: STAGMetadata) -> None:
    """验证 ID 范围和官方平衡标记的基本一致性。"""

    checks = (
        ("object_id", metadata.object_id, len(metadata.objects)),
        ("split_id", metadata.split_id, len(metadata.splits)),
    )
    for name, values, upper_bound in checks:
        if values.min() < 0 or values.max() >= upper_bound:
            raise ValueError(
                f"{name} 超出 [0, {upper_bound - 1}]："
                f"{values.min()}..{values.max()}"
            )
    if not np.all(metadata.has_valid_label[metadata.is_balanced]):
        raise ValueError("官方 isBalanced 样本中出现 hasValidLabel=False。")


def build_frame_splits(
    metadata: STAGMetadata,
    excluded_class: str = "empty_hand",
) -> FrameSplits:
    """使用官方平衡标记构造 26 类单帧训练与验证索引。

    官方 ``test`` 在本项目 notebook 中按用户要求作为逐 epoch 验证集使用。
    """

    try:
        excluded_id = metadata.objects.index(excluded_class)
        train_split_id = metadata.splits.index("train")
        validation_split_id = metadata.splits.index("test")
    except ValueError as error:
        raise ValueError("数据字典缺少 empty_hand、train 或 test。") from error

    original_object_ids = tuple(
        object_id
        for object_id in range(len(metadata.objects))
        if object_id != excluded_id
    )
    class_names = tuple(metadata.objects[i] for i in original_object_ids)
    original_to_label = np.full(len(metadata.objects), -1, dtype=np.int64)
    for label, object_id in enumerate(original_object_ids):
        original_to_label[object_id] = label

    eligible = metadata.is_balanced & (metadata.object_id != excluded_id)
    train_indices = np.flatnonzero(
        eligible & (metadata.split_id == train_split_id)
    ).astype(np.int64)
    validation_indices = np.flatnonzero(
        eligible & (metadata.split_id == validation_split_id)
    ).astype(np.int64)

    _validate_frame_splits(
        metadata,
        train_indices,
        validation_indices,
        original_to_label,
        len(class_names),
    )
    return FrameSplits(
        train_indices=train_indices,
        validation_indices=validation_indices,
        class_names=class_names,
        original_object_ids=original_object_ids,
        original_to_label=original_to_label,
    )


def _validate_frame_splits(
    metadata: STAGMetadata,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    original_to_label: np.ndarray,
    num_classes: int,
) -> None:
    """检查类别平衡、样本数和 recording 隔离。"""

    train_labels = original_to_label[metadata.object_id[train_indices]]
    validation_labels = original_to_label[
        metadata.object_id[validation_indices]
    ]
    if np.any(train_labels < 0) or np.any(validation_labels < 0):
        raise ValueError("划分中仍包含被排除类别。")

    train_counts = np.bincount(train_labels, minlength=num_classes)
    validation_counts = np.bincount(
        validation_labels, minlength=num_classes
    )
    if not np.all(train_counts == 1_353):
        raise ValueError(f"训练集类别计数不符合官方平衡划分：{train_counts}")
    if not np.all(validation_counts == 597):
        raise ValueError(
            f"验证集类别计数不符合官方平衡划分：{validation_counts}"
        )

    def recordings(indices: np.ndarray) -> set[tuple[int, int]]:
        return set(
            zip(
                metadata.batch_id[indices].tolist(),
                metadata.recording_id[indices].tolist(),
            )
        )

    overlap = recordings(train_indices) & recordings(validation_indices)
    if overlap:
        raise ValueError(f"训练与验证 recording 发生重叠：{sorted(overlap)}")


class STAGSingleFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """从内存缓存返回一张归一化压力图及其类别标签。

    Dataset 初始化时会一次性矢量化处理全部选中帧。训练期间的
    ``__getitem__`` 只建立 Tensor 视图，不再逐样本转换和归一化。
    """

    def __init__(
        self,
        metadata: STAGMetadata,
        frame_indices: np.ndarray,
        original_to_label: np.ndarray,
        pressure_min: float = 500.0,
        pressure_max: float = 650.0,
    ) -> None:
        if pressure_max <= pressure_min:
            raise ValueError("pressure_max 必须大于 pressure_min。")
        if len(frame_indices) == 0:
            raise ValueError("frame_indices 不能为空。")

        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.original_to_label = np.asarray(
            original_to_label, dtype=np.int64
        )
        self.labels = self.original_to_label[
            metadata.object_id[self.frame_indices]
        ]
        if np.any(self.labels < 0):
            raise ValueError("Dataset 中包含未映射类别。")

        # 一次性缓存连续 float32 数组，避免每个 epoch 重复进行逐样本归一化。
        images = metadata.pressure[self.frame_indices].astype(
            np.float32, copy=True
        )
        images -= float(pressure_min)
        images /= float(pressure_max - pressure_min)
        np.clip(images, 0.0, 1.0, out=images)
        images *= metadata.sensor_mask[None, :, :]
        self.images = np.ascontiguousarray(images)

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).unsqueeze(0)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, label

    @property
    def cache_size_bytes(self) -> int:
        """返回归一化图像缓存占用的字节数。"""

        return int(self.images.nbytes)


def create_single_frame_datasets(
    zip_path: str | Path,
) -> tuple[
    STAGMetadata,
    FrameSplits,
    STAGSingleFrameDataset,
    STAGSingleFrameDataset,
]:
    """读取真实数据并返回 metadata、划分、训练集和验证集。"""

    metadata = load_stag_metadata(zip_path)
    splits = build_frame_splits(metadata)
    train_dataset = STAGSingleFrameDataset(
        metadata, splits.train_indices, splits.original_to_label
    )
    validation_dataset = STAGSingleFrameDataset(
        metadata, splits.validation_indices, splits.original_to_label
    )
    return metadata, splits, train_dataset, validation_dataset
