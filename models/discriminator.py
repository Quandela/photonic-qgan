import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, image_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # self.model.to(torch.float64)

    def forward(self, x):
        # x.to(torch.float64)
        # print(x)
        # print(x.dtype)
        return self.model(x)