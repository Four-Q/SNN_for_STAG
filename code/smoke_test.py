"""真实 STAG 数据与 Conv-SNN 的非正式训练冒烟测试。

该脚本只使用两个样本完成一次前向和反向，不遍历训练集、不执行任何 epoch，
也不会在 outputs 中保存模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from spikingjelly.activation_based import functional
from torch import nn
from torch.utils.data import DataLoader, Subset

from snn_model import ConvSNNClassifier
from stag_data import (
    build_window_index,
    load_stag_metadata,
    validate_window_manifest,
)
from stag_dataset import STAGSequenceDataset, compute_class_weights
from train_utils import (
    EarlyStopping,
    evaluate,
    set_random_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    default_zip = (
        Path(__file__).resolve().parent.parent
        / "stag_data"
        / "classification_lite.zip"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=default_zip,
        help="classification_lite.zip 路径",
    )
    return parser.parse_args()


def check_early_stopping() -> None:
    """覆盖首次提升、准确率同分损失降低和 patience 用尽。"""

    stopper = EarlyStopping(patience=2)
    improved, should_stop = stopper.step(0.50, 1.00, epoch=1)
    assert improved and not should_stop and stopper.best_epoch == 1

    improved, should_stop = stopper.step(0.50, 0.90, epoch=2)
    assert improved and not should_stop and stopper.best_epoch == 2

    improved, should_stop = stopper.step(0.49, 0.80, epoch=3)
    assert not improved and not should_stop
    improved, should_stop = stopper.step(0.48, 0.70, epoch=4)
    assert not improved and should_stop


def main() -> None:
    args = parse_args()
    set_random_seed(42)

    print("1/5 读取 metadata.mat 并推导传感器掩码……")
    metadata = load_stag_metadata(args.zip_path)
    assert metadata.num_frames == 135_187
    assert metadata.num_recordings == 82
    assert metadata.num_objects == 27
    assert int(metadata.sensor_mask.sum()) == 548

    print("2/5 构建并逐窗口验证默认时序清单……")
    window_index = build_window_index(
        metadata,
        window_size=16,
        stride=8,
        mode="interaction",
        min_valid_ratio=0.5,
    )
    train_manifest = window_index.split_manifest("train")
    test_manifest = window_index.split_manifest("test")
    assert len(window_index.class_names) == 26
    assert len(train_manifest) == 8_315
    assert len(test_manifest) == 3_352
    assert train_manifest["recording_id"].nunique() == 52
    assert test_manifest["recording_id"].nunique() == 26
    assert set(train_manifest["recording_id"]).isdisjoint(
        set(test_manifest["recording_id"])
    )
    validate_window_manifest(metadata, window_index)

    print("3/5 读取两个真实训练窗口……")
    train_dataset = STAGSequenceDataset(
        metadata,
        train_manifest,
        window_index.recording_indices,
    )
    # 只取两个样本，确保 train_one_epoch 只运行一个批次。
    tiny_dataset = Subset(train_dataset, [0, 1])
    loader = DataLoader(tiny_dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert tuple(batch["x"].shape) == (2, 16, 1, 32, 32)
    assert tuple(batch["y"].shape) == (2,)
    assert torch.isfinite(batch["x"]).all()
    assert batch["x"].min() >= 0 and batch["x"].max() <= 1

    print("4/5 执行一次 Conv-SNN 前向、反向、评估和参数更新……")
    model = ConvSNNClassifier(num_classes=26)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    class_weights = compute_class_weights(train_manifest, 26)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 先明确检查公开 forward 接口的输出形状。
    with torch.no_grad():
        try:
            logits = model(batch["x"])
        finally:
            functional.reset_net(model)
    assert tuple(logits.shape) == (2, 26)

    train_result = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        num_classes=26,
        show_progress=False,
    )
    test_result = evaluate(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        num_classes=26,
        show_progress=False,
    )
    assert np_is_finite(train_result.loss)
    assert np_is_finite(test_result.loss)
    assert train_result.confusion_matrix.shape == (26, 26)
    assert test_result.confusion_matrix.shape == (26, 26)

    # 再检查一次手动重置不会报错。
    try:
        _ = model(batch["x"])
    finally:
        functional.reset_net(model)

    print("5/5 验证早停状态机……")
    check_early_stopping()
    print(
        "冒烟测试通过：真实数据读取、8315/3352 窗口、模型前向/反向、"
        "状态重置和早停逻辑均正常。"
    )


def np_is_finite(value: float) -> bool:
    """避免仅为一个标量额外导入 NumPy。"""

    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    main()
