"""无界面执行单帧 SNN notebook 的 smoke 模式并验证输出产物。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
from matplotlib import image as mpimg
from nbclient import NotebookClient


EXPECTED_FILES = (
    "best_model.pt",
    "last_model.pt",
    "history.csv",
    "summary.json",
    "class_mapping.json",
    "training_curves.png",
    "confusion_matrix.png",
    "class_accuracy.png",
)


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    code_dir = Path(__file__).resolve().parent
    notebook_path = code_dir / "train_single_frame_snn.ipynb"
    if not notebook_path.is_file():
        raise FileNotFoundError(f"找不到 notebook：{notebook_path}")

    with tempfile.TemporaryDirectory(prefix="stag_single_frame_smoke_") as tmp:
        temp_dir = Path(tmp)
        old_cwd = Path.cwd()
        old_run_mode = os.environ.get("SNN_RUN_MODE")
        old_output_root = os.environ.get("SNN_OUTPUT_ROOT")
        os.environ["SNN_RUN_MODE"] = "smoke"
        os.environ["SNN_OUTPUT_ROOT"] = str(temp_dir)

        try:
            os.chdir(code_dir)
            notebook = nbformat.read(notebook_path, as_version=4)
            notebook_source = "\n".join(
                cell.source
                for cell in notebook.cells
                if cell.cell_type == "code"
            )
            assert "from tqdm.auto import tqdm" in notebook_source
            assert "progress_bar = tqdm(" in notebook_source
            assert 'description=f"训练 {epoch}/{EPOCHS}"' in notebook_source
            assert 'description=f"验证 {epoch}/{EPOCHS}"' in notebook_source
            executed = NotebookClient(
                notebook,
                timeout=600,
                kernel_name="python3",
                allow_errors=False,
            ).execute()
            nbformat.write(executed, temp_dir / "executed_smoke.ipynb")
        finally:
            os.chdir(old_cwd)
            _restore_environment("SNN_RUN_MODE", old_run_mode)
            _restore_environment("SNN_OUTPUT_ROOT", old_output_root)

        output_dir = temp_dir / "single_frame_smoke"
        missing = [
            name for name in EXPECTED_FILES if not (output_dir / name).is_file()
        ]
        if missing:
            raise AssertionError(f"smoke 模式缺少输出文件：{missing}")

        summary = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["run_mode"] == "smoke"
        assert summary["num_frames"] == 135_187
        assert summary["num_classes"] == 26
        assert summary["full_train_samples"] == 35_178
        assert summary["full_validation_samples"] == 15_522
        assert summary["validation_uses_official_test"] is True
        assert "选择偏差" in summary["validation_warning"]
        assert summary["performance_config"]["batch_size"] == 4
        assert summary["performance_config"]["validation_batch_size"] == 4
        assert summary["performance_config"]["num_workers"] == 0
        assert summary["performance_config"]["amp_dtype"] is None
        assert summary["performance_config"]["fused_adam"] is False

        mapping = json.loads(
            (output_dir / "class_mapping.json").read_text(encoding="utf-8")
        )
        assert len(mapping) == 26
        assert all(row["name"] != "empty_hand" for row in mapping)

        history = pd.read_csv(output_dir / "history.csv")
        assert len(history) == 1
        numeric = history.select_dtypes(include=[np.number]).to_numpy()
        assert np.isfinite(numeric).all()

        for checkpoint_name in ("best_model.pt", "last_model.pt"):
            checkpoint = torch.load(
                output_dir / checkpoint_name,
                map_location="cpu",
                weights_only=False,
            )
            assert checkpoint["model_config"]["num_classes"] == 26
            assert checkpoint["class_names"] == summary["class_names"]

        for image_name in (
            "training_curves.png",
            "confusion_matrix.png",
            "class_accuracy.png",
        ):
            image = mpimg.imread(output_dir / image_name)
            assert image.ndim in (2, 3)
            assert image.shape[0] > 10 and image.shape[1] > 10
            assert np.isfinite(image).all()

    print(
        "冒烟测试通过：Notebook 无界面执行、真实 STAG 数据、一次训练/验证、"
        "模型文件、指标文件和英文图表均正常。"
    )


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


if __name__ == "__main__":
    main()
