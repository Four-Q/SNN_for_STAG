"""基于 SpikingJelly 的 STAG 多步卷积脉冲神经网络。"""

from __future__ import annotations

import torch
from torch import nn
from spikingjelly.activation_based import layer, neuron, surrogate


class ConvSNNClassifier(nn.Module):
    """使用连续触觉帧进行物体分类的轻量 Conv-SNN。

    输入
    ----
    x:
        ``[batch, time, channel, height, width]``，默认通道数为 1。

    输出
    ----
    ``[batch, num_classes]`` 的分类 logits。

    网络隐藏层使用 LIF 脉冲神经元。线性读出层保留为模拟值，并对所有真实触觉
    时间步求平均；这样既保留 SNN 的状态与脉冲计算，又能稳定使用交叉熵训练。
    """

    def __init__(
        self,
        num_classes: int = 26,
        in_channels: int = 1,
        channels: tuple[int, int, int] = (16, 32, 64),
        tau: float = 2.0,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes 必须大于 1。")
        if len(channels) != 3 or any(channel <= 0 for channel in channels):
            raise ValueError("channels 必须包含三个正整数。")

        c1, c2, c3 = channels

        def lif_node() -> neuron.LIFNode:
            return neuron.LIFNode(
                tau=tau,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                step_mode="m",
                backend="torch",
            )

        # SpikingJelly 的多步层约定输入首维为时间：[T, B, ...]。
        self.features = nn.Sequential(
            layer.Conv2d(
                in_channels,
                c1,
                kernel_size=3,
                padding=1,
                bias=False,
                step_mode="m",
            ),
            layer.BatchNorm2d(c1, step_mode="m"),
            lif_node(),
            layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m"),
            layer.Conv2d(
                c1,
                c2,
                kernel_size=3,
                padding=1,
                bias=False,
                step_mode="m",
            ),
            layer.BatchNorm2d(c2, step_mode="m"),
            lif_node(),
            layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m"),
            layer.Conv2d(
                c2,
                c3,
                kernel_size=3,
                padding=1,
                bias=False,
                step_mode="m",
            ),
            layer.BatchNorm2d(c3, step_mode="m"),
            lif_node(),
            layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m"),
        )
        self.spatial_pool = layer.AdaptiveAvgPool2d(
            output_size=(1, 1),
            step_mode="m",
        )
        self.classifier = layer.Linear(
            c3,
            num_classes,
            step_mode="m",
        )

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.channels = tuple(int(value) for value in channels)
        self.tau = float(tau)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用适合卷积网络的清晰、可复现初始化策略。"""

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, layer.Conv2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
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

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(
                "模型输入应为 [B, T, C, H, W]，"
                f"实际得到 {tuple(x.shape)}。"
            )
        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"模型需要 {self.in_channels} 个输入通道，"
                f"实际得到 {x.shape[2]}。"
            )

        # [B, T, C, H, W] -> [T, B, C, H, W]
        x_seq = x.permute(1, 0, 2, 3, 4).contiguous()
        features = self.features(x_seq)
        features = self.spatial_pool(features)
        features = torch.flatten(features, start_dim=2)
        logits_sequence = self.classifier(features)
        logits = logits_sequence.mean(dim=0)

        if return_sequence:
            return logits, logits_sequence
        return logits

    def get_config(self) -> dict[str, object]:
        """返回可写入 checkpoint/config.json 的模型配置。"""

        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "channels": list(self.channels),
            "tau": self.tau,
        }


def count_trainable_parameters(model: nn.Module) -> int:
    """返回需要梯度的模型参数数量。"""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

