"""使用直接电流编码的轻量 STAG 单帧卷积脉冲神经网络。"""

from __future__ import annotations

import torch
from spikingjelly.activation_based import layer, neuron, surrogate
from torch import nn


class DirectCurrentEncoder(nn.Module):
    """将一张归一化压力图重复为多个 SNN 内部仿真时间步。

    每个时间步接收完全相同的连续压力值。仿真时间步不代表连续采集的
    多张触觉帧，也不会像泊松编码一样随机生成二值脉冲。
    """

    def __init__(self, time_steps: int = 64) -> None:
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps 必须为正整数。")
        self.time_steps = int(time_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "直接电流编码器输入应为 [B, C, H, W]，"
                f"实际得到 {tuple(x.shape)}。"
            )
        return x.unsqueeze(0).expand(self.time_steps, *x.shape)


class SingleFrameConvSNN(nn.Module):
    """用内部直接电流时间步识别一张 32×32 压力图。

    输入是 ``[B, 1, 32, 32]``，并不是触觉帧序列。模型先将单张压力图
    重复为 ``[T, B, 1, 32, 32]`` 的直接电流，再由多步 LIF 网络计算。
    """

    def __init__(
        self,
        num_classes: int = 26,
        time_steps: int = 64,
        channels: tuple[int, int, int] = (16, 32, 64),
        hidden_features: int = 32,
        tau: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes 必须大于 1。")
        if len(channels) != 3 or any(value <= 0 for value in channels):
            raise ValueError("channels 必须包含三个正整数。")
        if hidden_features <= 0:
            raise ValueError("hidden_features 必须为正整数。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)。")

        c1, c2, c3 = channels

        def lif() -> neuron.LIFNode:
            return neuron.LIFNode(
                tau=tau,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                step_mode="m",
                backend="torch",
            )

        self.encoder = DirectCurrentEncoder(time_steps)
        self.features = nn.Sequential(
            layer.Conv2d(
                1, c1, kernel_size=3, padding=1, bias=False, step_mode="m"
            ),
            layer.BatchNorm2d(c1, step_mode="m"),
            lif(),
            layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m"),
            layer.Conv2d(
                c1, c2, kernel_size=3, padding=1, bias=False, step_mode="m"
            ),
            layer.BatchNorm2d(c2, step_mode="m"),
            lif(),
            layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m"),
            layer.Dropout2d(p=dropout, step_mode="m"),
            layer.Conv2d(
                c2, c3, kernel_size=3, padding=1, bias=False, step_mode="m"
            ),
            layer.BatchNorm2d(c3, step_mode="m"),
            lif(),
            layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m"),
        )
        self.spatial_pool = layer.AdaptiveAvgPool2d(
            output_size=(1, 1), step_mode="m"
        )
        self.hidden_classifier = layer.Linear(
            c3, hidden_features, step_mode="m"
        )
        self.hidden_lif = lif()
        self.classifier = layer.Linear(
            hidden_features, num_classes, step_mode="m"
        )

        self.num_classes = int(num_classes)
        self.time_steps = int(time_steps)
        self.channels = tuple(int(value) for value in channels)
        self.hidden_features = int(hidden_features)
        self.tau = float(tau)
        self.dropout = float(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, layer.Conv2d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, layer.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Linear, layer.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "模型输入应为 [B, 1, 32, 32]，"
                f"实际得到 {tuple(x.shape)}。"
            )
        if x.shape[1:] != (1, 32, 32):
            raise ValueError(
                "模型需要单通道 32×32 压力图，"
                f"实际得到 {tuple(x.shape[1:])}。"
            )

        current_by_step = self.encoder(x)
        features = self.features(current_by_step)
        features = self.spatial_pool(features)
        features = torch.flatten(features, start_dim=2)
        hidden_by_step = self.hidden_lif(
            self.hidden_classifier(features)
        )
        logits_by_step = self.classifier(hidden_by_step)
        return logits_by_step.mean(dim=0)

    def get_config(self) -> dict[str, object]:
        return {
            "num_classes": self.num_classes,
            "time_steps": self.time_steps,
            "channels": list(self.channels),
            "hidden_features": self.hidden_features,
            "tau": self.tau,
            "dropout": self.dropout,
            "input_encoding": "direct_current",
        }


def count_trainable_parameters(model: nn.Module) -> int:
    """返回可训练参数数量。"""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
