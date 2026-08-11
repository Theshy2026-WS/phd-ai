import torch
import torch.nn as nn


def make_activation(name):
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()](inplace=True) if name.lower() == "relu" else activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported activation '{name}'. Choose from: {', '.join(activations)}"
        ) from exc


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> activation, configured as one building block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation="relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """A basic ResNet block containing two convolutional layers."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation="relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        self.activation = make_activation(activation)
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.activation(self.residual(x) + self.shortcut(x))


class ECAAttention(nn.Module):
    """Efficient Channel Attention with a small 1D channel convolution."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be a positive odd number")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.pool(x)
        weights = weights.squeeze(-1).transpose(-1, -2)
        weights = self.channel_conv(weights)
        weights = weights.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class SelfAttention2D(nn.Module):
    """Global self-attention that preserves a BCHW feature-map interface."""

    def __init__(
        self,
        channels,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
        fusion="residual",
        gate_init=-2.0,
        positional_kernel_size=3,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if fusion not in {"direct", "residual", "gated"}:
            raise ValueError(
                "fusion must be one of: direct, residual, gated"
            )
        if positional_kernel_size <= 0 or positional_kernel_size % 2 == 0:
            raise ValueError(
                "positional_kernel_size must be a positive odd number"
            )

        hidden_dim = int(channels * mlp_ratio)
        self.fusion = fusion
        self.position_embedding = nn.Conv2d(
            channels,
            channels,
            kernel_size=positional_kernel_size,
            padding=positional_kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )
        self.gate = (
            nn.Parameter(torch.tensor(float(gate_init)))
            if fusion == "gated"
            else None
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        positioned = x + self.position_embedding(x)
        tokens = positioned.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        attended = self.attention_dropout(attended)

        if self.fusion == "direct":
            fused = attended
        elif self.fusion == "gated":
            fused = tokens + torch.sigmoid(self.gate) * attended
        else:
            fused = tokens + attended

        fused = fused + self.mlp(self.norm2(fused))
        return fused.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )


BLOCK_REGISTRY = {
    "conv": ConvBlock,
    "residual": ResidualBlock,
}


def build_2d_architecture(architecture, input_channels, activation="relu"):
    """Build an ordered BCHW architecture from YAML-compatible specs."""

    layers = []
    current_channels = input_channels

    for index, raw_spec in enumerate(architecture, start=1):
        spec = dict(raw_spec)
        try:
            block_type = spec.pop("type").lower()
        except KeyError as exc:
            raise ValueError(
                f"Architecture block at position {index} has no type"
            ) from exc
        out_channels = spec.pop("out_channels", current_channels)

        if block_type in {"eca", "self_attention_2d"}:
            if out_channels != current_channels:
                raise ValueError(
                    f"{block_type} block at position {index} cannot change "
                    "channel count"
                )
            block_class = (
                ECAAttention if block_type == "eca" else SelfAttention2D
            )
            layers.append(block_class(channels=current_channels, **spec))
            continue

        try:
            block_class = BLOCK_REGISTRY[block_type]
        except KeyError as exc:
            supported = ", ".join(
                [*BLOCK_REGISTRY, "eca", "self_attention_2d"]
            )
            raise ValueError(
                f"Unsupported architecture block '{block_type}' at position "
                f"{index}. Choose from: {supported}"
            ) from exc

        block_activation = spec.pop("activation", activation)
        layers.append(
            block_class(
                in_channels=current_channels,
                out_channels=out_channels,
                activation=block_activation,
                **spec,
            )
        )
        current_channels = out_channels

    return nn.Sequential(*layers), current_channels


class CNN(nn.Module):
    """Build a CNN from an ordered list of YAML module specifications."""

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        architecture=None,
        activation="relu",
        dropout=0.3,
    ):
        super().__init__()

        if architecture is None:
            architecture = [
                {"type": "conv", "out_channels": 32, "stride": 2},
                {"type": "conv", "out_channels": 64, "stride": 2},
                {"type": "conv", "out_channels": 128, "stride": 2},
            ]

        self.features, current_channels = build_2d_architecture(
            architecture=architecture,
            input_channels=input_channels,
            activation=activation,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
