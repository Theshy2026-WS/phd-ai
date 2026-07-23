import torch
import torch.nn as nn

from models.cnn import ConvBlock, ECAAttention, ResidualBlock
from models.vit import TransformerBlock


class CNNViTHybrid(nn.Module):
    """CNN local feature extraction followed by global Transformer blocks."""

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        image_size=32,
        cnn_architecture=None,
        activation="relu",
        embed_dim=256,
        transformer_depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.0,
    ):
        super().__init__()
        if cnn_architecture is None:
            raise ValueError("cnn_architecture must contain at least one CNN block")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        cnn_layers = []
        current_channels = input_channels
        feature_size = image_size

        for index, raw_spec in enumerate(cnn_architecture, start=1):
            spec = dict(raw_spec)
            block_type = spec.pop("type").lower()
            out_channels = spec.pop("out_channels", current_channels)
            block_activation = spec.pop("activation", activation)

            if block_type == "eca":
                if out_channels != current_channels:
                    raise ValueError(
                        f"ECA block at position {index} cannot change channel count"
                    )
                cnn_layers.append(
                    ECAAttention(
                        channels=current_channels,
                        **spec,
                    )
                )
                continue

            stride = spec.get("stride", 1)
            if block_type == "conv":
                block_class = ConvBlock
            elif block_type == "residual":
                block_class = ResidualBlock
            else:
                raise ValueError(
                    f"Unsupported CNN block '{block_type}' at position {index}"
                )

            cnn_layers.append(
                block_class(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    activation=block_activation,
                    **spec,
                )
            )
            current_channels = out_channels
            feature_size = (feature_size + stride - 1) // stride

        self.cnn_features = nn.Sequential(*cnn_layers)
        self.feature_projection = (
            nn.Identity()
            if current_channels == embed_dim
            else nn.Conv2d(current_channels, embed_dim, kernel_size=1)
        )

        self.num_tokens = feature_size ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens + 1, embed_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(transformer_depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.cnn_features(x)
        x = self.feature_projection(x)
        x = x.flatten(2).transpose(1, 2)

        if x.size(1) != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} CNN feature tokens, got {x.size(1)}"
            )

        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.embedding_dropout(x + self.position_embedding)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])
