import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from pycocotools.coco import COCO


class COCODataset(data.Dataset):
    def __init__(self, root, images_dir, annotation_path, is_categorical=False, transforms=None, instance_seg=False):
        self.root = root
        self.images_dir = images_dir
        self.coco = COCO(annotation_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.is_categorical = is_categorical
        self.transforms = transforms
        self.instance_seg = instance_seg
        self.num_classes = 92  # Background included

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_ids)
        img_dict = coco.loadImgs(img_id)[0]
        img_fn = img_dict['file_name']
        img = Image.open(os.path.join(self.images_dir, img_fn)).convert('RGB')
        height = int(img_dict['height'])
        width = int(img_dict['width'])

        num_objs = len(ann)

        areas = []
        boxes = []
        category_names = []
        labels = []

        for i in range(num_objs):
            x_min = ann[i]['bbox'][0]
            y_min = ann[i]['bbox'][1]
            x_max = x_min + ann[i]['bbox'][2]
            y_max = y_min + ann[i]['bbox'][3]
            boxes.append([y_min, x_min, y_max, x_max])
            areas.append(ann[i]['area'])

            category_id = ann[i]['category_id']
            labels.append(category_id)
            category_names.append(coco.loadCats(category_id)[0]['name'])

        if self.is_categorical:
            labels = self.to_categorical(labels, self.num_classes)

        if len(ann) > 0:
            masks = coco.annToMask(ann[0])
            for i in range(1, num_objs):
                masks = masks | coco.annToMask(ann[i])
        else:
            masks = []

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        my_annotation = {}
        my_annotation['height'] = height
        my_annotation['width'] = width
        my_annotation['mask'] = masks
        my_annotation['bbox'] = boxes
        my_annotation['label'] = labels
        my_annotation['image_id'] = img_id
        my_annotation['area'] = areas
        my_annotation['iscrowd'] = iscrowd
        my_annotation['category_name'] = category_names
        my_annotation['file_name'] = img_fn

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def to_categorical(label, num_classes):
        label_list = []
        if isinstance(label, list):
            for l in label:
                label_base = [0 for _ in range(num_classes)]
                label_base[l] = 1
                label_list.append(label_base)
        else:
            label_base = [0 for _ in range(num_classes)]
            label_base[label] = 1
            label_list.append(label_base)

        return label_list

    @staticmethod
    def to_categorical_multi_label(label, num_classes):
        label_result = [0 for _ in range(num_classes)]
        if isinstance(label, list):
            label_result
            for l in label:
                label_result[l] = 1
        else:
            label_result[label] = 1

        return label_result


def custom_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # print('data: ', data)
    # print('target: ', target)
    return [data, target]
