"""基于 Nature 修改版 ResNet-18 拓扑的单帧脉冲神经网络。

2019 年 Nature STAG 论文使用的是普通 CNN。本文件保留论文描述的
ResNet-18 空间拓扑，并将 ReLU 激活替换为 LIF 脉冲神经元，因此属于
SNN 工程改造版，而不是论文中不存在的“原始 SNN”。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from spikingjelly.activation_based import layer, neuron, surrogate
from torch import nn


class DirectCurrentEncoder(nn.Module):
    """将一张静态压力图重复为多个 SNN 内部仿真时间步。

    编码后的张量形状为 ``[T, B, C, H, W]``。每个时间步接收相同的
    连续电流值；这里的 ``T`` 不代表连续采集的多张触觉帧。
    """

    def __init__(self, time_steps: int = 16) -> None:
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps 必须为正整数。")
        self.time_steps = int(time_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "直接电流编码器需要 [B, C, H, W] 输入，"
                f"实际收到 {tuple(x.shape)}。"
            )
        return x.unsqueeze(0).expand(self.time_steps, *x.shape)


def _make_lif(tau: float) -> neuron.LIFNode:
    """创建适用于多步传播的 LIF 神经元。"""

    return neuron.LIFNode(
        tau=tau,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode="m",
        backend="torch",
    )


class SpikingBasicBlock(nn.Module):
    """ResNet-18 BasicBlock 的脉冲化实现。"""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        lif_factory: Callable[[], neuron.LIFNode],
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("BasicBlock 的 stride 只能为 1 或 2。")

        self.conv1 = layer.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            step_mode="m",
        )
        self.bn1 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif1 = lif_factory()
        self.conv2 = layer.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            step_mode="m",
        )
        self.bn2 = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif2 = lif_factory()

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                layer.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    step_mode="m",
                ),
                layer.BatchNorm2d(out_channels, step_mode="m"),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.lif2(out + identity)
        return out


class SpikingNatureResNet18(nn.Module):
    """用于 32×32 STAG 单帧物体分类的脉冲化 ResNet-18。

    空间结构遵循 Nature 论文的修改：

    - 首层卷积改为 3×3、stride=1；
    - 仅保留 ResNet-18 的前两个 layer group；
    - 两个 layer group 之间使用 20% 空间 dropout；
    - 最终空间特征维度为 128。

    输入为 ``[B, 1, 32, 32]``，返回 ``[B, num_classes]``。
    """

    def __init__(
        self,
        num_classes: int = 26,
        time_steps: int = 16,
        tau: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes 必须大于 1。")
        if tau <= 1.0:
            raise ValueError("LIF 神经元的 tau 必须大于 1。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 范围。")

        self.num_classes = int(num_classes)
        self.time_steps = int(time_steps)
        self.tau = float(tau)
        self.dropout = float(dropout)
        self.encoder = DirectCurrentEncoder(time_steps=self.time_steps)

        lif_factory = lambda: _make_lif(self.tau)
        self.stem = nn.Sequential(
            layer.Conv2d(
                1,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                step_mode="m",
            ),
            layer.BatchNorm2d(64, step_mode="m"),
            lif_factory(),
            layer.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                step_mode="m",
            ),
        )
        self.layer1 = self._make_layer(
            in_channels=64,
            out_channels=64,
            blocks=2,
            first_stride=1,
            lif_factory=lif_factory,
        )
        self.spatial_dropout = layer.Dropout2d(
            p=self.dropout,
            step_mode="m",
        )
        self.layer2 = self._make_layer(
            in_channels=64,
            out_channels=128,
            blocks=2,
            first_stride=2,
            lif_factory=lif_factory,
        )
        self.spatial_pool = layer.AdaptiveAvgPool2d(
            output_size=(1, 1),
            step_mode="m",
        )
        self.classifier = layer.Linear(
            128,
            self.num_classes,
            step_mode="m",
        )
        self._initialize_weights()

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        first_stride: int,
        lif_factory: Callable[[], neuron.LIFNode],
    ) -> nn.Sequential:
        if blocks <= 0:
            raise ValueError("每个 ResNet layer group 至少需要一个 block。")
        modules: list[nn.Module] = [
            SpikingBasicBlock(
                in_channels,
                out_channels,
                first_stride,
                lif_factory,
            )
        ]
        modules.extend(
            SpikingBasicBlock(
                out_channels,
                out_channels,
                1,
                lif_factory,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*modules)

    def _initialize_weights(self) -> None:
        """使用与 ResNet 常见实现一致的初始化策略。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "模型需要 [B, 1, 32, 32] 输入，"
                f"实际收到 {tuple(x.shape)}。"
            )
        if tuple(x.shape[1:]) != (1, 32, 32):
            raise ValueError(
                "模型仅接受单通道 32×32 STAG 压力图，"
                f"实际收到 {tuple(x.shape[1:])}。"
            )

        current_by_step = self.encoder(x)
        features = self.stem(current_by_step)
        features = self.layer1(features)
        features = self.spatial_dropout(features)
        features = self.layer2(features)
        features = self.spatial_pool(features)
        features = torch.flatten(features, start_dim=2)
        logits_by_step = self.classifier(features)
        return logits_by_step.mean(dim=0)

    def get_config(self) -> dict[str, object]:
        """返回可用于重建模型的配置。"""

        return {
            "architecture": "spiking_nature_resnet18",
            "num_classes": self.num_classes,
            "time_steps": self.time_steps,
            "tau": self.tau,
            "dropout": self.dropout,
            "input_encoding": "direct_current",
            "input_shape": [1, 32, 32],
            "layer_groups": [2, 2],
            "channels": [64, 128],
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, object],
    ) -> SpikingNatureResNet18:
        """从 ``get_config`` 或 checkpoint 中保存的配置重建模型。"""

        expected_architecture = "spiking_nature_resnet18"
        architecture = config.get("architecture", expected_architecture)
        if architecture != expected_architecture:
            raise ValueError(
                f"无法用本类重建 architecture={architecture!r} 的模型。"
            )
        return cls(
            num_classes=int(config["num_classes"]),
            time_steps=int(config["time_steps"]),
            tau=float(config["tau"]),
            dropout=float(config["dropout"]),
        )


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中需要梯度的参数数量。"""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
