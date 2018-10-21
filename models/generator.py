import torch
import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.linear1 = nn.Linear(self.latent_dim, 128)
        self.lr1 = nn.LeakyReLU(0.2)

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, 0.8)
        self.lr2 = nn.LeakyReLU(0.2)

        self.linear3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512, 0.8)
        self.lr3 = nn.LeakyReLU(0.2)

        self.linear4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024, 0.8)
        self.lr4 = nn.LeakyReLU(0.2)

        self.out = nn.Linear(1024, int(np.prod(image_shape)))

    def forward(self, z):
        out1 = self.lr1(self.linear1(z))

        out2 = self.lr2(self.bn2(self.linear2(out1)))

        out3 = self.lr3(self.bn3(self.linear3(out2)))

        out4 = self.lr4(self.bn4(self.linear4(out3)))

        logit = self.out(out4)

        out = torch.tanh(logit)

        return logit, out