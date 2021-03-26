import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image

if __name__ == '__main__':
    a = (-30, 30)
    b = np.random.randint(a[0], a[1])
    print(b)