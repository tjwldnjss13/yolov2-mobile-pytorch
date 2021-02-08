import torch.nn as nn

from models.dsconv import DSConv


class Darknet19DS(nn.Module):
    def __init__(self, num_classes):
        super(Darknet19DS, self).__init__()
        self.num_classes = num_classes
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.models = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            self.maxpool,
            DSConv(32, 64, 3, 1, 1, True),
            self.maxpool,
            DSConv(64, 128, 3, 1, 1, True),
            DSConv(128, 64, 1, 1, 0, True),
            DSConv(64, 128, 3, 1, 1, True),
            self.maxpool,
            DSConv(128, 256, 3, 1, 1, True),
            DSConv(256, 128, 1, 1, 0, True),
            DSConv(128, 256, 3, 1, 1, True),
            self.maxpool,
            DSConv(256, 512, 3, 1, 1, True),
            DSConv(512, 256, 1, 1, 0, True),
            DSConv(256, 512, 3, 1, 1, True),
            DSConv(512, 256, 1, 1, 0, True),
            DSConv(256, 512, 3, 1, 1, True),
            self.maxpool,
            DSConv(512, 1024, 3, 1, 1, True),
            DSConv(1024, 512, 1, 1, 0, True),
            DSConv(512, 1024, 3, 1, 1, True),
            DSConv(1024, 512, 1, 1, 0, True),
            DSConv(512, 1024, 3, 1, 1, True),
        )

    def forward(self, x):
        x = self.models(x)

        return x

