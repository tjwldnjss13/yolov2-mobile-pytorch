import torch.nn as nn


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
        self.classifier = nn.Sequential(
            DSConv(1024, self.num_classes, 1, 1, 0, True),
            self.avgpool,
        )
        self.reg_layer = DSConv(1024, 36, 3, 1, 1, True)

    def forward(self, x):
        x = self.models(x)

        # x = self.classifier(x)

        x = self.reg_layer(x)

        return x


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=False):
        super(DSConv, self).__init__()
        self.batch_norm = batch_norm
        self.dconv = DConv(in_channels, kernel_size, stride, padding)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU(True)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.conv1x1.weight)

    def forward(self, x):
        x = self.dconv(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        return x


class DConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(DConv, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.dconv.weight)

    def forward(self, x):
        return self.dconv(x)
