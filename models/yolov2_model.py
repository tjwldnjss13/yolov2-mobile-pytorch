import torch
import torch.nn as nn

from models.darknet19_ds import Darknet19DS
from models.dsconv import DSConv
from utils.rpn_util import generate_anchor_box


class YOLOV2Mobile(nn.Module):
    def __init__(self, in_size, num_classes, anchor_box_samples):
        # Inputs:
        #    in_size: tuple or list, (input height, input width)
        #    num_classes: int
        #    anchor_box_samples: tensor, [# anchor box, (y1, x1, y2, x2)]

        super(YOLOV2Mobile, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.anchor_box_samples = anchor_box_samples

        self.feature_model = Darknet19DS(self.num_classes)
        self.reg_layer = DSConv(1024, len(anchor_box_samples) * (self.num_classes + 5), 1, 1, 0, True)

        self.anchor_boxes = self._get_valid_anchor_boxes()

    def _get_valid_anchor_boxes(self):
        dummy = torch.zeros(1, 3, 416, 416)
        out_dummy = self.feature_model(dummy)
        h_out = out_dummy.shape[2]
        anchor_stride = int(self.in_size[0] / h_out)
        anchor_boxes, valid_mask = generate_anchor_box(self.anchor_box_samples, self.in_size, anchor_stride)

        return anchor_boxes[torch.nonzero(valid_mask, as_tuple=False)].squeeze(1)

    def forward(self, x):
        x = self.feature_model(x)
        x = self.reg_layer(x)

        return x

