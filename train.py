import os
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from dataset.coco_dataset import COCODataset, custom_collate_fn
from models.yolov2_model import YOLOV2Mobile
from loss import yolo_custom_loss


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--lr', type=float, required=False, default=.001)
    parser.add_argument('--weight_decay', type=float, required=False, default=.005)
    parser.add_argument('--num_epochs', type=int, required=False, default=100)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs

    # Generate COCO dataset
    root = 'D://DeepLearningData/COCOdataset2017/'
    img_dir = None
    ann_pth = None
    transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
    dset = COCODataset(root, img_dir, transforms=transform)

    # Load model
    model = YOLOV2Mobile(in_size=(416, 416), num_classes=91, anchor_box_samples=None).to(device)

    # Define optimizer, loss function
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = yolo_custom_loss

























