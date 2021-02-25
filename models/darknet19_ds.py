import torch
import torch.nn as nn

from models.dsconv import DSConv


activation_dict = {'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(True), 'leaky_relu': nn.LeakyReLU(inplace=True), 'relu6': nn.ReLU6(True)}


class Darknet19DS(nn.Module):
    def __init__(self):
        super(Darknet19DS, self).__init__()
        self.activation_ds = 'relu6'
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            DSConv(32, 64, 3, 1, 1, self.activation_ds),
            nn.MaxPool2d(2, 2),

            DSConv(64, 128, 3, 1, 1, self.activation_ds),
            # DSConv(128, 64, 1, 1, 0, self.activation_ds),
            nn.Conv2d(128, 64, 1, 1, 0),
            activation_dict['relu6'],
            DSConv(64, 128, 3, 1, 1, self.activation_ds),
            nn.MaxPool2d(2, 2),

            DSConv(128, 256, 3, 1, 1, self.activation_ds),
            # DSConv(256, 128, 1, 1, 0, self.activation_ds),
            nn.Conv2d(256, 128, 1, 1, 0),
            activation_dict['relu6'],
            DSConv(128, 256, 3, 1, 1, self.activation_ds),
            nn.MaxPool2d(2, 2),

            DSConv(256, 512, 3, 1, 1, self.activation_ds),
            # DSConv(512, 256, 1, 1, 0, self.activation_ds),
            nn.Conv2d(512, 256, 1, 1, 0),
            activation_dict['relu6'],
            DSConv(256, 512, 3, 1, 1, self.activation_ds),
            # DSConv(512, 256, 1, 1, 0, self.activation_ds),
            nn.Conv2d(512, 256, 1, 1, 0),
            activation_dict['relu6'],
            DSConv(256, 512, 3, 1, 1, self.activation_ds),
        )
        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(2, 2),

            DSConv(512, 1024, 3, 1, 1, self.activation_ds),
            # DSConv(1024, 512, 1, 1, 0, self.activation_ds),
            nn.Conv2d(1024, 512, 1, 1, 0),
            activation_dict['relu6'],
            DSConv(512, 1024, 3, 1, 1, self.activation_ds),
            # DSConv(1024, 512, 1, 1, 0, self.activation_ds),
            nn.Conv2d(1024, 512, 1, 1, 0),
            activation_dict['relu6'],
            DSConv(512, 1024, 3, 1, 1, self.activation_ds),
            DSConv(1024, 1024, 3, 1, 1, self.activation_ds),
            DSConv(1024, 1024, 3, 1, 1, self.activation_ds),
            DSConv(1024, 1024, 3, 1, 1, self.activation_ds),
        )
        # self.layer_3 = DSConv(3072, 1024, 1, 1, 0, self.activation_ds)
        self.layer_3 = nn.Sequential(
            nn.Conv2d(3072, 1024, 1, 1, 0),
            activation_dict['relu6'],
        )
        self.passthrough_layer = DSConv(512, 512, 3, 1, 1, self.activation_ds)

    def forward(self, x):
        _pass = x = self.layer_1(x)
        x = self.layer_2(x)

        _pass = self.passthrough_layer(_pass)
        h, w = _pass.shape[2:]
        h_cut, w_cut = int(h / 2), int(w / 2)
        _pass = torch.cat([_pass[:, :, :h_cut, :w_cut],
                           _pass[:, :, :h_cut, w_cut:],
                           _pass[:, :, h_cut:, :w_cut],
                           _pass[:, :, h_cut:, w_cut:]], dim=1)

        x = torch.cat([x, _pass], dim=1)
        # x = self.activation(x)

        x = self.layer_3(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = Darknet19DS().cuda()
    summary(model, (3, 416, 416))

















