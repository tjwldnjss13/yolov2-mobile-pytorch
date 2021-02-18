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

a = torch.Tensor([[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]],
                  [[11, 22, 33],
                   [44, 55, 66],
                   [77, 88, 99]]])
print(a.shape)
b = torch.zeros(2, 3)

for i in range(a.shape[0]):
    a[i] *= b[i]
print(a)


