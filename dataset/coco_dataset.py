import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from pycocotools.coco import COCO

import os
from PIL import Image
import matplotlib.pyplot as plt


class COCODataset(data.Dataset):
    def __init__(self, root, annotation, categorical=True, transforms=None, instance_seg=False):
        self.root = root
        self.annotation = annotation
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.categorical = categorical
        self.transforms = transforms
        self.instance_seg = instance_seg

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_ids)
        img = coco.loadImgs(img_id)[0]
        img_path = img['file_name']
        img_return = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        num_objs = len(ann)

        areas = []
        boxes = []
        cats = []
        labels = []
        for i in range(num_objs):
            x_min = ann[i]['bbox'][0]
            y_min = ann[i]['bbox'][1]
            x_max = x_min + ann[i]['bbox'][2]
            y_max = y_min + ann[i]['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])
            areas.append(ann[i]['area'])
            if self.instance_seg:
                labels.append(ann[i]['category_id'])
            else:
                labels.append(1)
            cat_id = ann[i]['category_id']
            cats.append(coco.loadCats(cat_id)[0]['name'])

        if len(ann) > 0:
            masks = coco.annToMask(ann[0])
            for i in range(1, num_objs):
                masks = masks | coco.annToMask(ann[i])
        else:
            masks = []

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.ones((num_objs, ), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        my_annotation = {}
        my_annotation['mask'] = masks
        my_annotation['bbox'] = boxes
        my_annotation['label'] = labels
        my_annotation['image_id'] = img_id
        my_annotation['area'] = areas
        my_annotation['iscrowd'] = iscrowd
        my_annotation['category'] = cats
        my_annotation['file_name'] = img_path

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def to_categorical(labels, n_label):
    labels_cate = []
    for label in labels:
        temp = [0 for _ in range(n_label)]
        temp[label] = 1
        labels_cate.append(temp)

    labels_cate = torch.Tensor(labels_cate)

    return labels_cate


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # print('data: ', data)
    # print('target: ', target)
    return [data, target]
