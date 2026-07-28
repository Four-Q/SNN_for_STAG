"""STAG 元数据读取、连续片段检测与时序窗口清单构建。

本模块只依赖 NumPy、Pandas 和 SciPy，不依赖 PyTorch。这样数据探索 notebook
可以独立运行，也便于先检查数据是否正确，再进入模型训练阶段。

设计要点：
1. 直接从 classification_lite.zip 中读取 metadata.mat，不在磁盘上解压。
2. 始终以 (batchId, recordingId) 为连续记录的唯一标识。
3. 先尊重官方 splitId，再在每条 recording 内生成窗口。
4. 窗口清单只保存索引和元数据，不复制压力帧。
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat


REQUIRED_MAT_FIELDS = (
    "recordings",
    "recordingId",
    "objects",
    "objectId",
    "pressure",
    "frame",
    "ts",
    "batchId",
    "splitId",
    "splits",
    "batches",
    "hasValidLabel",
    "isBalanced",
    "isGrasp",
    "isTransition",
)


@dataclass(slots=True)
class STAGMetadata:
    """内存中的 STAG metadata.mat。

    数值字段的第 0 维都与 ``pressure`` 的帧维一一对应。字符串表使用元组保存，
    对应的 ``*_id`` 保留官方的零起始索引。
    """

    pressure: np.ndarray
    frame: np.ndarray
    timestamp: np.ndarray
    batch_id: np.ndarray
    recording_id: np.ndarray
    object_id: np.ndarray
    split_id: np.ndarray
    has_valid_label: np.ndarray
    is_balanced: np.ndarray
    is_grasp: np.ndarray
    is_transition: np.ndarray
    recordings: tuple[str, ...]
    objects: tuple[str, ...]
    splits: tuple[str, ...]
    batches: tuple[str, ...]
    sensor_mask: np.ndarray

    @property
    def num_frames(self) -> int:
        """总帧数。"""

        return int(self.pressure.shape[0])

    @property
    def num_recordings(self) -> int:
        """recording 名称数量。"""

        return len(self.recordings)

    @property
    def num_objects(self) -> int:
        """原始类别数量（包含 empty_hand）。"""

        return len(self.objects)


@dataclass(slots=True)
class STAGWindowIndex:
    """窗口清单以及从 recording 偏移量回到原始帧的索引。

    ``manifest`` 中不保存压力数据。Dataset 使用 ``recording_indices`` 找到某条
    recording 排序后的原始帧下标，再根据 ``start_offset`` 动态切片。
    """

    manifest: pd.DataFrame
    recording_indices: dict[tuple[int, int], np.ndarray]
    class_names: tuple[str, ...]
    original_object_id_to_label: dict[int, int]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def split_manifest(self, split: str) -> pd.DataFrame:
        """返回指定官方划分的独立 DataFrame。"""

        result = self.manifest.loc[self.manifest["split"] == split].copy()
        return result.reset_index(drop=True)


def inspect_zip(zip_path: str | Path) -> pd.DataFrame:
    """只查看 ZIP 内文件名和大小，不解压文件。"""

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        rows = [
            {
                "name": item.filename,
                "size_bytes": item.file_size,
                "compressed_bytes": item.compress_size,
            }
            for item in archive.infolist()
        ]
    return pd.DataFrame(rows)


def inspect_mat_variables(zip_path: str | Path) -> pd.DataFrame:
    """返回 metadata.mat 中变量的名称、形状和 MATLAB 类型。"""

    with zipfile.ZipFile(Path(zip_path)) as archive:
        with archive.open("metadata.mat") as mat_file:
            variables = whosmat(mat_file)
    return pd.DataFrame(variables, columns=["name", "shape", "matlab_type"])


def read_embedded_readme(zip_path: str | Path) -> str:
    """读取 ZIP 中官方 readme.html 的文本内容。"""

    with zipfile.ZipFile(Path(zip_path)) as archive:
        return archive.read("readme.html").decode("utf-8", errors="replace")


def _matlab_string_list(value: np.ndarray) -> tuple[str, ...]:
    """把 scipy.loadmat 返回的 cell/string 数组转换为 Python 字符串元组。"""

    strings: list[str] = []
    for item in np.atleast_1d(value).ravel():
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        if isinstance(item, bytes):
            strings.append(item.decode("utf-8", errors="replace"))
        else:
            strings.append(str(item))
    return tuple(strings)


def infer_sensor_mask(
    pressure: np.ndarray,
    expected_sensor_count: int = 548,
) -> np.ndarray:
    """根据整个数据集中是否存在时间变化推导有效传感器位置。

    该 lite 数据包的 476 个填充位置始终等于 510，548 个真实传感器位置会随
    时间变化。因此 ``最大值 != 最小值`` 可以恢复官方描述的 548 点掩码。
    若未来数据包不符合这一性质，函数会明确报错，避免静默使用错误掩码。
    """

    if pressure.ndim != 3 or pressure.shape[1:] != (32, 32):
        raise ValueError(
            f"pressure 应为 [N, 32, 32]，实际得到 {pressure.shape}。"
        )

    sensor_mask = pressure.max(axis=0) != pressure.min(axis=0)
    actual_count = int(sensor_mask.sum())
    if actual_count != expected_sensor_count:
        raise ValueError(
            "无法可靠推导 STAG 传感器掩码："
            f"期望 {expected_sensor_count} 个变化位置，实际为 {actual_count}。"
        )
    return sensor_mask


def load_stag_metadata(
    zip_path: str | Path,
    expected_sensor_count: int = 548,
) -> STAGMetadata:
    """从 ZIP 内存读取 metadata.mat，并完成基本一致性检查。"""

    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"找不到 STAG 压缩包：{zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if "metadata.mat" not in names:
            raise KeyError(f"{zip_path} 中不存在 metadata.mat。")
        # BytesIO 让 scipy 直接从内存解析，不在工作区生成解压文件。
        mat_buffer = io.BytesIO(archive.read("metadata.mat"))

    raw = loadmat(
        mat_buffer,
        variable_names=list(REQUIRED_MAT_FIELDS),
        squeeze_me=True,
    )
    missing = [field for field in REQUIRED_MAT_FIELDS if field not in raw]
    if missing:
        raise KeyError(f"metadata.mat 缺少字段：{missing}")

    pressure = np.asarray(raw["pressure"])
    frame_count = pressure.shape[0]

    def vector(name: str, dtype: Any) -> np.ndarray:
        result = np.asarray(raw[name], dtype=dtype).reshape(-1)
        if len(result) != frame_count:
            raise ValueError(
                f"字段 {name} 长度为 {len(result)}，与压力帧数 {frame_count} 不一致。"
            )
        return result

    metadata = STAGMetadata(
        pressure=pressure,
        frame=vector("frame", np.int64),
        timestamp=vector("ts", np.float64),
        batch_id=vector("batchId", np.int64),
        recording_id=vector("recordingId", np.int64),
        object_id=vector("objectId", np.int64),
        split_id=vector("splitId", np.int64),
        has_valid_label=vector("hasValidLabel", np.bool_),
        is_balanced=vector("isBalanced", np.bool_),
        is_grasp=vector("isGrasp", np.bool_),
        is_transition=vector("isTransition", np.bool_),
        recordings=_matlab_string_list(raw["recordings"]),
        objects=_matlab_string_list(raw["objects"]),
        splits=_matlab_string_list(raw["splits"]),
        batches=_matlab_string_list(raw["batches"]),
        sensor_mask=infer_sensor_mask(pressure, expected_sensor_count),
    )
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata: STAGMetadata) -> None:
    """检查 ID 范围、矩阵形状和基础字段一致性。"""

    if metadata.pressure.shape[1:] != (32, 32):
        raise ValueError(f"压力帧空间形状不是 32×32：{metadata.pressure.shape}")
    if not np.issubdtype(metadata.pressure.dtype, np.integer):
        raise TypeError(f"原始压力应为整数，实际为 {metadata.pressure.dtype}。")

    checks = (
        ("batch_id", metadata.batch_id, len(metadata.batches)),
        ("recording_id", metadata.recording_id, len(metadata.recordings)),
        ("object_id", metadata.object_id, len(metadata.objects)),
        ("split_id", metadata.split_id, len(metadata.splits)),
    )
    for name, values, upper_bound in checks:
        if values.min() < 0 or values.max() >= upper_bound:
            raise ValueError(
                f"{name} 超出合法范围 [0, {upper_bound - 1}]："
                f"最小 {values.min()}，最大 {values.max()}。"
            )


def build_recording_indices(
    metadata: STAGMetadata,
) -> dict[tuple[int, int], np.ndarray]:
    """按 ``timestamp → frame`` 为每条 recording 建立排序后的全局帧索引。"""

    recording_indices: dict[tuple[int, int], list[int]] = {}
    for global_index, key in enumerate(
        zip(metadata.batch_id.tolist(), metadata.recording_id.tolist())
    ):
        recording_indices.setdefault((int(key[0]), int(key[1])), []).append(
            global_index
        )

    sorted_indices: dict[tuple[int, int], np.ndarray] = {}
    for key, indices_list in recording_indices.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        order = np.lexsort(
            (metadata.frame[indices], metadata.timestamp[indices])
        )
        sorted_indices[key] = indices[order]
    return sorted_indices


def find_continuous_chunks(
    metadata: STAGMetadata,
    ordered_indices: np.ndarray,
    bad_pressure_threshold: float = 950.0,
    gap_factor: float = 2.5,
) -> list[tuple[int, int]]:
    """返回一条 recording 内可安全生成窗口的 ``[start, end)`` 偏移区间。

    异常帧自身不会进入任何 chunk，其前后也不会被重新拼接。
    """

    if len(ordered_indices) == 0:
        return []

    frames = metadata.frame[ordered_indices]
    timestamps = metadata.timestamp[ordered_indices]
    pressure = metadata.pressure[ordered_indices]
    bad_frame = (pressure > bad_pressure_threshold).any(axis=(1, 2))

    frame_diff = np.diff(frames)
    time_diff = np.diff(timestamps)
    positive_dt = time_diff[time_diff > 0]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else np.inf
    pair_break = (
        (frame_diff != 1)
        | (time_diff <= 0)
        | (time_diff > gap_factor * median_dt)
    )

    chunks: list[tuple[int, int]] = []
    start = 0
    for position in range(len(ordered_indices)):
        # 异常帧前的正常区间先收尾，再从下一帧重新开始。
        if bad_frame[position]:
            if start < position:
                chunks.append((start, position))
            start = position + 1
            continue

        # pair_break[k] 表示第 k 帧与第 k+1 帧之间不可连续。
        if position > 0 and pair_break[position - 1]:
            if start < position:
                chunks.append((start, position))
            start = position

    if start < len(ordered_indices):
        chunks.append((start, len(ordered_indices)))
    return chunks


def _validate_recording_labels(
    metadata: STAGMetadata,
    key: tuple[int, int],
    ordered_indices: np.ndarray,
) -> tuple[int, int]:
    """确认 recording 内只有一个物体类别和一个官方划分。"""

    object_ids = np.unique(metadata.object_id[ordered_indices])
    split_ids = np.unique(metadata.split_id[ordered_indices])
    if len(object_ids) != 1:
        raise ValueError(f"recording {key} 包含多个 objectId：{object_ids}")
    if len(split_ids) != 1:
        raise ValueError(f"recording {key} 跨越多个 splitId：{split_ids}")
    return int(object_ids[0]), int(split_ids[0])


def build_window_index(
    metadata: STAGMetadata,
    window_size: int = 16,
    stride: int = 8,
    mode: str = "interaction",
    min_valid_ratio: float = 0.5,
    include_empty_hand: bool = False,
    bad_pressure_threshold: float = 950.0,
    gap_factor: float = 2.5,
) -> STAGWindowIndex:
    """生成不复制压力数据的连续时序窗口清单。

    Parameters
    ----------
    mode:
        ``"interaction"`` 保留有效比例不低于 ``min_valid_ratio`` 的窗口；
        ``"stable"`` 只保留全部帧均有效的窗口。
    """

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size 和 stride 必须为正整数。")
    if mode not in {"interaction", "stable"}:
        raise ValueError("mode 必须是 'interaction' 或 'stable'。")
    if not 0.0 <= min_valid_ratio <= 1.0:
        raise ValueError("min_valid_ratio 必须位于 [0, 1]。")

    kept_original_ids = [
        object_id
        for object_id, name in enumerate(metadata.objects)
        if include_empty_hand or name != "empty_hand"
    ]
    original_to_label = {
        original_id: label
        for label, original_id in enumerate(kept_original_ids)
    }
    class_names = tuple(metadata.objects[i] for i in kept_original_ids)
    recording_indices = build_recording_indices(metadata)

    rows: list[dict[str, Any]] = []
    for key in sorted(recording_indices):
        ordered = recording_indices[key]
        original_object_id, split_id = _validate_recording_labels(
            metadata, key, ordered
        )
        if original_object_id not in original_to_label:
            continue

        batch_id, recording_id = key
        chunks = find_continuous_chunks(
            metadata=metadata,
            ordered_indices=ordered,
            bad_pressure_threshold=bad_pressure_threshold,
            gap_factor=gap_factor,
        )
        for chunk_start, chunk_end in chunks:
            if chunk_end - chunk_start < window_size:
                continue
            last_start = chunk_end - window_size
            for start_offset in range(chunk_start, last_start + 1, stride):
                end_offset = start_offset + window_size
                window_indices = ordered[start_offset:end_offset]
                valid_ratio = float(
                    metadata.has_valid_label[window_indices].mean()
                )
                keep = (
                    np.isclose(valid_ratio, 1.0)
                    if mode == "stable"
                    else valid_ratio >= min_valid_ratio
                )
                if not keep:
                    continue

                label = original_to_label[original_object_id]
                rows.append(
                    {
                        "split": metadata.splits[split_id],
                        "split_id": split_id,
                        "batch_id": batch_id,
                        "batch_name": metadata.batches[batch_id],
                        "recording_id": recording_id,
                        "recording_name": metadata.recordings[recording_id],
                        "start_offset": start_offset,
                        "length": window_size,
                        "original_object_id": original_object_id,
                        "label": label,
                        "object_name": metadata.objects[original_object_id],
                        "valid_ratio": valid_ratio,
                        "start_frame": int(metadata.frame[window_indices[0]]),
                        "end_frame": int(metadata.frame[window_indices[-1]]),
                        "start_timestamp": float(
                            metadata.timestamp[window_indices[0]]
                        ),
                        "end_timestamp": float(
                            metadata.timestamp[window_indices[-1]]
                        ),
                    }
                )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("当前参数没有生成任何时序窗口。")
    manifest = manifest.sort_values(
        ["split_id", "batch_id", "recording_id", "start_offset"]
    ).reset_index(drop=True)

    return STAGWindowIndex(
        manifest=manifest,
        recording_indices=recording_indices,
        class_names=class_names,
        original_object_id_to_label=original_to_label,
    )


def build_recording_table(metadata: STAGMetadata) -> pd.DataFrame:
    """生成每条 recording 一行的概览表，供探索 notebook 使用。"""

    rows: list[dict[str, Any]] = []
    for key, ordered in build_recording_indices(metadata).items():
        object_id, split_id = _validate_recording_labels(metadata, key, ordered)
        dt = np.diff(metadata.timestamp[ordered])
        positive_dt = dt[dt > 0]
        rows.append(
            {
                "batch_id": key[0],
                "batch_name": metadata.batches[key[0]],
                "recording_id": key[1],
                "recording_name": metadata.recordings[key[1]],
                "object_id": object_id,
                "object_name": metadata.objects[object_id],
                "split": metadata.splits[split_id],
                "frame_count": len(ordered),
                "valid_ratio": float(
                    metadata.has_valid_label[ordered].mean()
                ),
                "duration_seconds": float(
                    metadata.timestamp[ordered[-1]]
                    - metadata.timestamp[ordered[0]]
                ),
                "median_dt": (
                    float(np.median(positive_dt))
                    if len(positive_dt)
                    else np.nan
                ),
                "bad_frame_count": int(
                    (
                        metadata.pressure[ordered] > 950
                    ).any(axis=(1, 2)).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["batch_id", "recording_id"]
    ).reset_index(drop=True)


def build_continuity_report(
    metadata: STAGMetadata,
    bad_pressure_threshold: float = 950.0,
    gap_factor: float = 2.5,
) -> pd.DataFrame:
    """统计每条 recording 的结构断点、异常帧和连续 chunk 数量。"""

    rows: list[dict[str, Any]] = []
    for key, ordered in build_recording_indices(metadata).items():
        frames = metadata.frame[ordered]
        timestamps = metadata.timestamp[ordered]
        dt = np.diff(timestamps)
        df = np.diff(frames)
        positive_dt = dt[dt > 0]
        median_dt = float(np.median(positive_dt)) if len(positive_dt) else np.inf
        structural_breaks = (
            (df != 1) | (dt <= 0) | (dt > gap_factor * median_dt)
        )
        bad_frames = (
            metadata.pressure[ordered] > bad_pressure_threshold
        ).any(axis=(1, 2))
        chunks = find_continuous_chunks(
            metadata,
            ordered,
            bad_pressure_threshold=bad_pressure_threshold,
            gap_factor=gap_factor,
        )
        rows.append(
            {
                "batch_id": key[0],
                "recording_id": key[1],
                "recording_name": metadata.recordings[key[1]],
                "structural_breaks": int(structural_breaks.sum()),
                "bad_frames": int(bad_frames.sum()),
                "continuous_chunks": len(chunks),
                "longest_chunk": max(
                    (end - start for start, end in chunks), default=0
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_window_manifest(
    metadata: STAGMetadata,
    window_index: STAGWindowIndex,
    bad_pressure_threshold: float = 950.0,
    gap_factor: float = 2.5,
) -> None:
    """逐窗口验证 recording、类别、帧号、时间和异常值边界。"""

    for row in window_index.manifest.itertuples(index=False):
        key = (int(row.batch_id), int(row.recording_id))
        ordered = window_index.recording_indices[key]
        start = int(row.start_offset)
        end = start + int(row.length)
        indices = ordered[start:end]

        if len(indices) != int(row.length):
            raise AssertionError(f"窗口 {key}/{start} 长度不足。")
        if len(np.unique(metadata.object_id[indices])) != 1:
            raise AssertionError(f"窗口 {key}/{start} 跨越物体类别。")
        if len(np.unique(metadata.split_id[indices])) != 1:
            raise AssertionError(f"窗口 {key}/{start} 跨越官方划分。")
        if not np.all(np.diff(metadata.frame[indices]) == 1):
            raise AssertionError(f"窗口 {key}/{start} 帧号不连续。")

        dt = np.diff(metadata.timestamp[indices])
        if not np.all(dt > 0):
            raise AssertionError(f"窗口 {key}/{start} 时间戳未严格递增。")
        recording_dt = np.diff(metadata.timestamp[ordered])
        positive_dt = recording_dt[recording_dt > 0]
        median_dt = float(np.median(positive_dt))
        if np.any(dt > gap_factor * median_dt):
            raise AssertionError(f"窗口 {key}/{start} 跨越时间跳跃。")
        if np.any(metadata.pressure[indices] > bad_pressure_threshold):
            raise AssertionError(f"窗口 {key}/{start} 含异常压力帧。")


def summarize_window_index(window_index: STAGWindowIndex) -> pd.DataFrame:
    """按划分汇总窗口、recording 和类别数量。"""

    return (
        window_index.manifest.groupby("split", sort=False)
        .agg(
            windows=("label", "size"),
            recordings=("recording_id", "nunique"),
            classes=("label", "nunique"),
            mean_valid_ratio=("valid_ratio", "mean"),
        )
        .reset_index()
    )


def save_class_mapping(
    path: str | Path,
    class_names: Iterable[str],
    original_object_id_to_label: Mapping[int, int],
) -> None:
    """保存模型标签、原始 STAG objectId 和类别名称的映射。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(class_names)
    label_to_original = {
        label: original_id
        for original_id, label in original_object_id_to_label.items()
    }
    rows = [
        {
            "label": label,
            "object_name": name,
            "original_object_id": int(label_to_original[label]),
        }
        for label, name in enumerate(names)
    ]
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

