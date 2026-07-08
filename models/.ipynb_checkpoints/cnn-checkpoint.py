import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, input_channels=1, num_classes=10):

        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels,
            32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(
            128 * 4 * 4,
            256
        )

        self.fc2 = nn.Linear(
            256,
            num_classes
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.pool(
            F.relu(self.conv1(x))
        )

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = self.pool(
            F.relu(self.conv3(x))
        )

        x = self.adaptive_pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.fc2(x)

        return x