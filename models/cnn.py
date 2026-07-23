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

        layers = []
        current_channels = input_channels

        for index, spec in enumerate(architecture, start=1):
            spec = dict(spec)
            block_type = spec.pop("type").lower()
            out_channels = spec.pop("out_channels", current_channels)
            block_activation = spec.pop("activation", activation)

            if block_type == "conv":
                block_class = ConvBlock
            elif block_type == "residual":
                block_class = ResidualBlock
            else:
                raise ValueError(
                    f"Unsupported architecture block '{block_type}' at position {index}"
                )

            layers.append(
                block_class(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    activation=block_activation,
                    **spec,
                )
            )
            current_channels = out_channels

        self.features = nn.Sequential(*layers)
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
