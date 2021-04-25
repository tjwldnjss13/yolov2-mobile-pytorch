import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.darknet53_ds import *

from PIL import Image

if __name__ == '__main__':
    a = torch.Tensor([float('nan'), 1, 2])
    for i in range(100):
        for j in range(100):
            for m in range(100):
                print(i, j, m)

                if torch.isnan(a).sum() > 0:
                    exit(0)