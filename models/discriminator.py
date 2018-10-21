import torch
import numpy as np
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(int(np.prod(image_shape)), 512)
        self.lr1 = nn.LeakyReLU(0.2)

        self.linear2 = nn.Linear(512, 256)
        self.lr2 = nn.LeakyReLU(0.2)

        self.out = nn.Linear(256, 1)

    def forward(self, image):
        flat_image = image.view(image.size(0), -1)
        l1 = self.lr1(self.linear1(flat_image))

        l2 = self.lr2(self.linear2(l1))

        logit = self.out(l2)

        validity = torch.sigmoid(logit)

        return logit, validity