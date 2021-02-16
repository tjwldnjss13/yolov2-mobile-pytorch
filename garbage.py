import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt

a = torch.Tensor([1.2])
print(int(a))

# boxes = [[ 49.6859, 139.2234, 135.7689, 300.9905],
#         [335.6590, 138.2104, 430.3932, 291.0888],
#         [142.9306, 165.9247, 311.0477, 348.2268],
#         [134.8581,  74.8888, 337.4425, 254.2883],
#         [209.4458, 233.1566, 254.0361, 281.3012]]
#
#
# img = np.zeros((500, 500, 3))
#
# cx, cy = 250, 250
#
# for box in boxes:
#     y1, x1, y2, x2 = box
#     y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
#     cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
#     h, w = y2 - y1, x2 - x1
#     y1, x1, y2, x2 = cy - int(.5 * h), cx - int(.5 * w), cy + int(.5 * h), cx + int(.5 * w)
#     cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=1)
#     print(h, w)
#
# plt.imshow(img)
# plt.show()
