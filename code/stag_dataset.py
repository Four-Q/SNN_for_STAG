"""把 STAG 窗口清单包装为 PyTorch Dataset。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stag_data import STAGMetadata


class STAGSequenceDataset(Dataset[dict[str, Any]]):
    """按窗口清单动态读取连续触觉帧。

    压力帧只在 ``__getitem__`` 时被切片和转为 float32，重叠窗口不会在内存或
    磁盘上复制保存。
    """

    def __init__(
        self,
        metadata: STAGMetadata,
        manifest: pd.DataFrame,
        recording_indices: dict[tuple[int, int], np.ndarray],
        pressure_min: float = 500.0,
        pressure_max: float = 650.0,
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
        self.sensor_mask = torch.from_numpy(
            metadata.sensor_mask.astype(np.bool_, copy=True)
        ).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.manifest)

    def _normalize_pressure(self, pressure: np.ndarray) -> np.ndarray:
        """把常用有效压力范围映射到 [0, 1]，再清零无效矩阵位置。"""

        x = pressure.astype(np.float32, copy=True)
        x -= self.pressure_min
        x /= self.pressure_max - self.pressure_min
        np.clip(x, 0.0, 1.0, out=x)
        x *= self.metadata.sensor_mask[None, :, :]
        return x

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        key = (int(row["batch_id"]), int(row["recording_id"]))
        ordered = self.recording_indices[key]
        start = int(row["start_offset"])
        length = int(row["length"])
        frame_indices = ordered[start : start + length]

        if len(frame_indices) != length:
            raise IndexError(f"窗口 {key}/{start} 越过 recording 末尾。")

        # 添加单通道维度，最终得到 [T, 1, 32, 32]。
        x = self._normalize_pressure(
            self.metadata.pressure[frame_indices]
        )[:, None, :, :]

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(int(row["label"]), dtype=torch.long),
            "valid_label": torch.from_numpy(
                self.metadata.has_valid_label[frame_indices].copy()
            ),
            "timestamps": torch.from_numpy(
                self.metadata.timestamp[frame_indices]
                .astype(np.float32, copy=True)
            ),
            "batch_id": int(row["batch_id"]),
            "recording_id": int(row["recording_id"]),
            "recording_name": str(row["recording_name"]),
            "start_frame": int(row["start_frame"]),
        }


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

