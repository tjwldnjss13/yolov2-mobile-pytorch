import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt


# boxes = [[130.2030,  84.8202, 331.2180, 259.0452],
#         [137.4449, 204.0276, 300.2263, 356.9088],
#         [369.0522, 250.4970, 433.3239, 328.3164],
#         [270.1581, 114.8591, 398.2650, 309.8760],
#         [ 50.1476, 135.2850, 139.3589, 302.8188]]
#
#
# img = np.zeros((500, 500, 3))
#
# cx, cy = 250, 250
#
# for box in boxes:
#     y1, x1, y2, x2 = box
#     y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
#     h, w = y2 - y1, x2 - x1
#     y1, x1, y2, x2 = cy - int(.5 * h), cx - int(.5 * w), cy + int(.5 * h), cx + int(.5 * w)
#     cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
#
# plt.imshow(img)
# plt.show()

from torchsummary import summary
from models.dsconv import DSConv


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1, 1)
        self.activation = nn.LeakyReLU(.01, True)

        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            self.activation,
        )

    def forward(self, x):
        # x = self.conv(x)
        # x = self.activation(x)

        x = self.layer(x)

        return x


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    from PIL import Image
    from dataset.augment import GaussianNoise

    img = Image.open('sample/boat.jpg')
    img = transforms.ToTensor()(img)
    plt.figure(0)
    plt.imshow(img.permute(1, 2, 0))

    def normalize(tensor):
        return TF.normalize(tensor, mean=[.485, .456, .406], std=[.229, .224, .225])

    img_norm = normalize(img)
    plt.figure(1)
    plt.imshow(img_norm.permute(1, 2, 0))

    img_noise = GaussianNoise(mean=0, std=.2)(img)
    plt.figure(2)
    plt.imshow(img_noise.permute(1, 2, 0))

    img_norm_noise = GaussianNoise(mean=0, std=.2)(img_norm)
    plt.figure(3)
    plt.imshow(img_norm_noise.permute(1, 2, 0))

    plt.show()

































