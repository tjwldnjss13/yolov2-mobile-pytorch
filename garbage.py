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
                  [[33, 22, 11],
                   [44, 66, 55],
                   [77, 88, 99]]])

idx = torch.max(a, dim=2).indices
# [0, 0, 0, 1, 1, 1] [0, 1, 2, 0, 1, 2] [2, 2, 2, 0, 1, 2]

idx1 = []
for i in range(idx.shape[0]):
    idx1 += [i for _ in range(idx.shape[1])]

idx2 = []
for i in range(idx.shape[0]):
    idx2 += [j for j in range(idx.shape[1])]

idx3 = idx.reshape(-1).squeeze()

print(a[idx1, idx2, idx3])
temp = torch.zeros(6)
a[idx1, idx2, idx3] = temp
print(a)



