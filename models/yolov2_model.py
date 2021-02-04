import torch
import torch.nn as nn

from models.rpn import RPN
from models.darknet19_ds import Darknet19DS
from utils.rpn_util import generate_anchor_box


class YOLOV2Mobile(nn.Module):
    def __init__(self, in_size, num_classes, anchor_box_scale, anchor_box_ratio):
        # Inputs:
        #    in_size: tuple or list (input height, input width)
        #    num_classes = int
        #    anchor_box_scale: list [scales]
        #    anchor_box_ratio: list [aspect ratios]

        super(YOLOV2Mobile, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.anchor_box_scale = anchor_box_scale
        self.anchor_box_ratio = anchor_box_ratio

        self.feature_model = Darknet19DS(self.num_classes)

        self.anchor_boxes = self._get_valid_anchor_boxes()

    def _get_valid_anchor_boxes(self):
        dummy = torch.zeros(1, 3, 416, 416)
        out_dummy = self.feature_model(dummy)
        h_out = out_dummy.shape[2]
        anchor_stride = int(self.in_size[0] / h_out)
        anchor_boxes, valid_mask = generate_anchor_box(self.anchor_box_ratio, self.anchor_box_scale, self.in_size, anchor_stride)

        return anchor_boxes[torch.nonzero(valid_mask, as_tuple=False)].squeeze(1)

    def forward(self, x):
        x = self.feature_model(x)

        x = x.reshape()


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

model = YOLOV2Mobile((448, 448), 20, [64, 238, 256], [.5, 1, 2])
anchor_boxes = model.anchor_boxes

# img = np.zeros((448, 448, 3))
# for box in anchor_boxes:
#     y1, x1, y2, x2 = box
#     y1, x1, y2, x2 = int(y1.item()), int(x1.item()), int(y2.item()), int(x2.item())
#     cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
#
# plt.imshow(img)
# plt.show()




























