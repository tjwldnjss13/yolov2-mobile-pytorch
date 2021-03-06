import torch
import torch.nn as nn

from models.backbone import YoloBackbone
from models.darknet19_ds import Darknet19DS
from models.darknet53_ds import Darknet53DS
from models.dsconv import DSConv
from utils.rpn_util import generate_anchor_box


class YOLOV2Mobile(nn.Module):
    def __init__(self, in_size, num_classes, anchor_box_samples):
        """
        :param in_size: tuple or list, (height of input, width of input)
        :param num_classes: int
        :param anchor_box_samples: tensor, [num anchor boxes, (height of anchor box, width of anchor box)]
        """

        super(YOLOV2Mobile, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.anchor_box_samples = torch.Tensor([[1.73145, 1.3221],
                                       [4.00944, 3.19275],
                                       [8.09892, 5.05587],
                                       [4.84053, 9.47112],
                                       [10.0071, 11.2364]])

        self.feature_model = YoloBackbone()
        # self.reg_layer = DSConv(1024, len(anchor_box_samples) * (5 + num_classes), 1, 1, 0, 'relu')
        self.reg_layer = nn.Sequential(
            nn.Conv2d(1024, len(anchor_box_samples) * (5 + num_classes), 1, 1)
        )

        # self.anchor_boxes = self._get_valid_anchor_boxes()

        self._initialize_weights()

    def _get_valid_anchor_boxes(self):
        dummy = torch.zeros(1, 3, 416, 416)
        out_dummy = self.feature_model(dummy)
        h_out = out_dummy.shape[2]
        anchor_stride = int(self.in_size[0] / h_out)
        anchor_boxes, valid_mask = generate_anchor_box(self.anchor_box_samples, self.in_size, anchor_stride)

        return anchor_boxes[torch.nonzero(valid_mask, as_tuple=False)].squeeze(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.feature_model(x)
        x = self.reg_layer(x)
        x = x.permute(0, 2, 3, 1)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = YOLOV2Mobile((416, 416), 20, torch.Tensor([[20, 30], [10, 40]])).cuda()
    summary(model, (3, 416, 416))
