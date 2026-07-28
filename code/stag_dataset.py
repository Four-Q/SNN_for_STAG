"""把 STAG 窗口清单包装为 PyTorch Dataset。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stag_data import STAGMetadata


def precompute_normalized_pressure(
    metadata: STAGMetadata,
    pressure_min: float = 500.0,
    pressure_max: float = 650.0,
) -> np.ndarray:
    """一次性归一化全部压力帧，避免重叠窗口重复执行相同计算。

    返回的 float32 数组会被训练集和测试集 Dataset 共享。STAG lite 数据集约
    占 0.52 GiB；在内存充足的训练主机上，这通常能显著减轻 DataLoader 的
    CPU 压力。
    """

    if pressure_max <= pressure_min:
        raise ValueError("pressure_max 必须大于 pressure_min。")
    x = metadata.pressure.astype(np.float32, copy=True)
    x -= float(pressure_min)
    x /= float(pressure_max - pressure_min)
    np.clip(x, 0.0, 1.0, out=x)
    x *= metadata.sensor_mask[None, :, :]
    return x


class STAGSequenceDataset(Dataset[dict[str, Any]]):
    """按窗口清单动态读取连续触觉帧。

    可传入一次性预归一化的共享压力数组；不传时，压力帧会在
    ``__getitem__`` 中按需转为 float32。窗口索引本身不会复制压力帧。
    """

    def __init__(
        self,
        metadata: STAGMetadata,
        manifest: pd.DataFrame,
        recording_indices: dict[tuple[int, int], np.ndarray],
        pressure_min: float = 500.0,
        pressure_max: float = 650.0,
        normalized_pressure: np.ndarray | None = None,
        include_metadata: bool = True,
    ) -> None:
        if pressure_max <= pressure_min:
            raise ValueError("pressure_max 必须大于 pressure_min。")
        if manifest.empty:
            raise ValueError("manifest 不能为空。")

        self.metadata = metadata
        self.manifest = manifest.reset_index(drop=True).copy()
        self.recording_indices = recording_indices
        self.pressure_min = float(pressure_min)
        self.pressure_max = float(pressure_max)
        self.include_metadata = bool(include_metadata)
        self.sensor_mask = torch.from_numpy(
            metadata.sensor_mask.astype(np.bool_, copy=True)
        ).unsqueeze(0)

        if normalized_pressure is not None:
            if normalized_pressure.shape != metadata.pressure.shape:
                raise ValueError(
                    "normalized_pressure 形状必须与 metadata.pressure 一致，"
                    f"实际为 {normalized_pressure.shape} 和 "
                    f"{metadata.pressure.shape}。"
                )
            if normalized_pressure.dtype != np.float32:
                raise TypeError("normalized_pressure 必须为 float32。")
        self.normalized_pressure = normalized_pressure

        # 把每个窗口的全局帧索引和常用标量提前缓存为紧凑数组。这样热路径不再
        # 逐样本调用 pandas.iloc、查字典并切片 recording 索引。
        frame_indices: list[np.ndarray] = []
        for batch_id, recording_id, start, length in self.manifest[
            ["batch_id", "recording_id", "start_offset", "length"]
        ].itertuples(index=False, name=None):
            key = (int(batch_id), int(recording_id))
            ordered = self.recording_indices[key]
            indices = ordered[int(start) : int(start) + int(length)]
            if len(indices) != int(length):
                raise IndexError(
                    f"窗口 {key}/{int(start)} 越过 recording 末尾。"
                )
            frame_indices.append(indices)
        self.frame_indices = np.stack(frame_indices)
        self.labels = self.manifest["label"].to_numpy(
            dtype=np.int64, copy=True
        )

        if self.include_metadata:
            self.batch_ids = self.manifest["batch_id"].to_numpy(
                dtype=np.int64, copy=True
            )
            self.recording_ids = self.manifest["recording_id"].to_numpy(
                dtype=np.int64, copy=True
            )
            self.recording_names = self.manifest[
                "recording_name"
            ].to_numpy(copy=True)
            self.start_frames = self.manifest["start_frame"].to_numpy(
                dtype=np.int64, copy=True
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def _normalize_pressure(self, pressure: np.ndarray) -> np.ndarray:
        """把常用有效压力范围映射到 [0, 1]，再清零无效矩阵位置。"""

        # frame_indices 的 NumPy 高级索引已经返回独立数组，无需再次强制复制。
        x = pressure.astype(np.float32, copy=False)
        x -= self.pressure_min
        x /= self.pressure_max - self.pressure_min
        np.clip(x, 0.0, 1.0, out=x)
        x *= self.metadata.sensor_mask[None, :, :]
        return x

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_indices = self.frame_indices[index]

        # 添加单通道维度，最终得到 [T, 1, 32, 32]。
        if self.normalized_pressure is None:
            x = self._normalize_pressure(
                self.metadata.pressure[frame_indices]
            )
        else:
            x = self.normalized_pressure[frame_indices]

        sample: dict[str, Any] = {
            "x": torch.from_numpy(x),
            "y": torch.tensor(self.labels[index], dtype=torch.long),
        }
        sample["x"] = sample["x"].unsqueeze(1)
        if not self.include_metadata:
            return sample

        sample.update(
            {
                "valid_label": torch.from_numpy(
                    self.metadata.has_valid_label[frame_indices].copy()
                ),
                "timestamps": torch.from_numpy(
                    self.metadata.timestamp[frame_indices].astype(
                        np.float32, copy=True
                    )
                ),
                "batch_id": int(self.batch_ids[index]),
                "recording_id": int(self.recording_ids[index]),
                "recording_name": str(self.recording_names[index]),
                "start_frame": int(self.start_frames[index]),
            }
        )
        return sample


def compute_class_weights(
    manifest: pd.DataFrame,
    num_classes: int,
) -> torch.Tensor:
    """根据训练窗口数计算均值归一化的类别反频率权重。"""

    counts = (
        manifest["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"训练清单缺少类别：{missing}")

    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
