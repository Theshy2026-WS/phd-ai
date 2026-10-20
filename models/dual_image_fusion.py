"""Independent raw-image branches; AACNN is a structure-only Tiny adaptation."""
import torch
from torch import nn

from models.vit import ViT
from models.attention_correction import state_digest


def ce(logits, targets):
    return -(targets * logits.log_softmax(-1)).sum(-1).mean()


class RawImageViT(ViT):
    def __init__(self):
        super().__init__(num_classes=200, image_size=64, patch_size=8,
                         embed_dim=192, depth=4, num_heads=4, mlp_ratio=2.,
                         dropout=.1)

    def features(self, x):
        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.class_token.expand(len(x), -1, -1), tokens), 1)
        return self.norm(self.blocks(self.embedding_dropout(
            tokens + self.position_embedding)))[:, 0]

    def forward(self, x):
        return self.classifier(self.features(x))


class DualImageFusion(nn.Module):
    """Frozen pretrained-on-Tiny CNN; trainable independent raw-image ViT."""
    def __init__(self, base, mode='prediction'):
        super().__init__()
        if mode not in ('prediction', 'feature'):
            raise ValueError(mode)
        self.mode = mode
        self.base = base.requires_grad_(False).eval()
        self.base_digest = state_digest(base)
        self.vit = RawImageViT()
        self.cnn_projection = nn.Sequential(nn.Linear(2048, 192), nn.LayerNorm(192))
        self.vit_projection = nn.LayerNorm(192)
        # Prediction gate reads normalized class distributions, not their scale.
        self.gate = nn.Sequential(nn.Linear(400 if mode == 'prediction' else 384, 64),
                                  nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        if mode == 'prediction':
            # Start conservative, but auxiliary CE keeps ViT learning.
            with torch.no_grad():
                self.gate[-1].bias.copy_(torch.tensor([2., 0.]))
            # Do not include unused projection parameters in this model.
            del self.cnn_projection, self.vit_projection
        else:
            self.classifier = nn.Linear(192, 200)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def verify_frozen(self):
        if state_digest(self.base) != self.base_digest:
            raise RuntimeError('Frozen CNN changed')

    def outputs(self, x):
        with torch.no_grad():
            f = self.base.stages(self.base.stem(x))
            h = self.base.head.gap(self.base.head.relu(self.base.head.bn(f))).flatten(1)
            cnn = self.base.head.dense(h)
        v = self.vit.features(x)  # Raw image, NOT CNN features.
        vit = self.vit.classifier(v)
        if self.mode == 'prediction':
            weights = self.gate(torch.cat((cnn.softmax(-1), vit.softmax(-1)), -1)).log_softmax(-1)
            mixed = torch.logsumexp(torch.stack((
                cnn.log_softmax(-1) + weights[:, :1],
                vit.log_softmax(-1) + weights[:, 1:]), dim=1), dim=1)
        else:
            c, v = self.cnn_projection(h), self.vit_projection(v)
            weights = self.gate(torch.cat((c, v), -1)).softmax(-1)
            mixed = self.classifier(weights[:, :1] * c + weights[:, 1:] * v)
        return mixed, vit

    def forward(self, x):
        return self.outputs(x)[0]

    def training_loss(self, x, targets):
        mixed, vit = self.outputs(x)
        # Explicit auxiliary objective prevents gate starvation of a random ViT.
        return ce(mixed, targets) + .5 * ce(vit, targets)


class AACNNTiny(nn.Module):
    """AACNN-ViT §4.4.3 gate topology, NOT its pretrained medical experiment."""
    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for out in (32, 64, 128):
            layers += [nn.Conv2d(channels, out, 3, padding=1), nn.ReLU(),
                       nn.BatchNorm2d(out), nn.MaxPool2d(2)]
            channels = out
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(128, 256), nn.ReLU())
        self.vit = RawImageViT()
        # AACNN uses a fused classifier, not a separate ViT classification head.
        self.vit.classifier = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(192, 256), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 2),
                                  nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(nn.Dropout(.4), nn.Linear(256, 256),
                                        nn.ReLU(), nn.Dropout(.3), nn.Linear(256, 200))

    def forward(self, x):
        c, v = self.cnn(x), self.projection(self.vit.features(x))
        w = self.gate(torch.cat((c, v), -1))
        return self.classifier(w[:, :1]*c + w[:, 1:]*v)
