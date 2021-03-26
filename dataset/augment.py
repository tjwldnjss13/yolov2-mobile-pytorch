import torch
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms

from PIL import Image


class GaussianNoise(object):
    def __init__(self, mean=0., std=.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}'


def rotate2d(image, bounding_box, angle):
    """
    :param image: Tensor, [channel, height, width]
    :param bounding_box: Tensor, [num bounding box, (y_min, x_min, y_max, x_max)]
    :param angle: int
    :return: img_rotate, bbox_rotate
    """
    _, h_og, w_og = image.shape
    img = image.permute(1, 2, 0).numpy()
    h, w, _ = img.shape
    x_ctr, y_ctr = int(w / 2), int(h / 2)

    bbox = bounding_box.numpy()

    mat = cv.getRotationMatrix2D((x_ctr, y_ctr), angle, 1)
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    mat[0, 2] += bound_w / 2 - x_ctr
    mat[1, 2] += bound_h / 2 - y_ctr

    img_rotate = cv.warpAffine(img, mat, (bound_w, bound_h))

    h_rotate, w_rotate, _ = img_rotate.shape
    x_ctr_rotate, y_ctr_rotate = int(w_rotate / 2), int(h_rotate / 2)

    theta = angle * np.pi / 180
    w_dif, h_dif = int((w_rotate - w) / 2), int((h_rotate - h) / 2)

    bbox_rotate_list = []
    if len(bbox) > 0:
        theta *= -1
        for i in range(len(bbox)):
            x0, y0, x2, y2 = bbox[i]
            x1, y1, x3, y3 = x2, y0, x0, y2
            x0, y0, x1, y1 = x0 + w_dif, y0 + h_dif, x1 + w_dif, y1 + h_dif
            x2, y2, x3, y3 = x2 + w_dif, y2 + h_dif, x3 + w_dif, y3 + h_dif

            x0_rot = int((((x0 - x_ctr_rotate) * np.cos(theta)) - ((y0 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y0_rot = int((((x0 - x_ctr_rotate) * np.sin(theta)) + ((y0 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x1_rot = int((((x1 - x_ctr_rotate) * np.cos(theta)) - ((y1 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y1_rot = int((((x1 - x_ctr_rotate) * np.sin(theta)) + ((y1 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x2_rot = int((((x2 - x_ctr_rotate) * np.cos(theta)) - ((y2 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y2_rot = int((((x2 - x_ctr_rotate) * np.sin(theta)) + ((y2 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x3_rot = int((((x3 - x_ctr_rotate) * np.cos(theta)) - ((y3 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y3_rot = int((((x3 - x_ctr_rotate) * np.sin(theta)) + ((y3 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))

            x_min, y_min = int(min(x0_rot, x1_rot, x2_rot, x3_rot)), int(min(y0_rot, y1_rot, y2_rot, y3_rot))
            x_max, y_max = int(max(x0_rot, x1_rot, x2_rot, x3_rot)), int(max(y0_rot, y1_rot, y2_rot, y3_rot))

            bbox_rotate_list.append([y_min, x_min, y_max, x_max])

    h_rot, w_rot, _ = img_rotate.shape
    h_ratio, w_ratio = h_og / h_rot, w_og / w_rot

    img_rotate = cv.resize(img_rotate, (w_og, h_og), interpolation=cv.INTER_CUBIC)

    img_rotate = torch.as_tensor(img_rotate).permute(2, 0, 1)
    bbox_rotate = torch.as_tensor(bbox_rotate_list).type(dtype=torch.float64)

    if len(bbox_rotate) > 0:
        bbox_rotate[:, 0] *= h_ratio
        bbox_rotate[:, 1] *= w_ratio
        bbox_rotate[:, 2] *= h_ratio
        bbox_rotate[:, 3] *= w_ratio

    return img_rotate, bbox_rotate


def horizontal_flip(image, bounding_box):
    """
    :param image: Tensor, [channel, height, width]
    :param bounding_box: Tensor, [num bounding box, (y_min, x_min, y_max, x_max)]
    :return:
    """
    img_flip = transforms.RandomHorizontalFlip(1)(image)
    _, h, w = img_flip.shape

    bbox = bounding_box
    for i in range(len(bbox)):
        bbox[i, 1], bbox[i, 3] = w - bbox[i, 3], w - bbox[i, 1]

    return img_flip, bbox


if __name__ == '__main__':
    img_pth = '../sample/boat.jpg'
    img = Image.open(img_pth)
    import torchvision.transforms as transforms
    img = transforms.ToTensor()(img)
    bbox = torch.Tensor([[150, 110, 240, 410]])

    img_flip, bbox_flip = horizontal_flip(img, bbox)

    print(bbox_flip)

    import matplotlib.pyplot as plt
    plt.imshow(img_flip.permute(1, 2, 0))
    plt.show()