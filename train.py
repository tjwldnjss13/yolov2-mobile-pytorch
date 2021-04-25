import os
import copy
import cv2 as cv
import numpy as np
import time
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset, Subset
from PIL import Image

from utils.util import time_calculator
from utils.pytorch_util import make_batch
from utils.yolov2_tensor_generator import get_output_anchor_box_tensor, get_yolo_v2_output_tensor, get_yolo_v2_target_tensor
from dataset.coco_dataset import COCODataset, custom_collate_fn
# from dataset.voc_dataset import VOCDataset, custom_collate_fn
from dataset.augment import GaussianNoise
from models.yolov2_model import YOLOV2Mobile
from loss import yolov2_custom_loss_1 as loss_func
from early_stopping import EarlyStopping


def get_coco_dataset(root):
    from dataset.coco_dataset import COCODataset, custom_collate_fn
    dset_name = 'coco2017'
    root = 'D://DeepLearningData/COCOdataset2017/'
    train_img_dir = os.path.join(root, 'images', 'train')
    val_img_dir = os.path.join(root, 'images', 'val')
    train_ann_pth = os.path.join(root, 'annotations', 'instances_train2017.json')
    val_ann_pth = os.path.join(root, 'annotations', 'instances_val2017.json')
    transform_og = transforms.Compose([transforms.Resize((416, 416)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_noise = transforms.Compose([transforms.Resize((416, 416)),
                                          transforms.ToTensor(),
                                          GaussianNoise(mean=0, std=.2),
                                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    train_dset = COCODataset(root=root, images_dir=train_img_dir, annotation_path=train_ann_pth, image_size=(416, 416),
                             is_categorical=True, transforms=transform_og)
    train_dset_noise = COCODataset(root=root, images_dir=train_img_dir, annotation_path=train_ann_pth, image_size=(416, 416),
                                   is_categorical=True, transforms=transform_noise)
    # train_dset_rotate = COCODataset(root=root, images_dir=train_img_dir, annotation_path=train_ann_pth, image_size=(416, 416), is_categorical=True, transforms=transform_og, rotate_angle=(-30, 30))

    num_classes = train_dset.num_classes

    # train_dset = ConcatDataset([train_dset, train_dset_noise])
    val_dset = COCODataset(root=root, images_dir=val_img_dir, annotation_path=val_ann_pth, image_size=(416, 416),
                           is_categorical=True, transforms=transform_og)

    collate_fn = custom_collate_fn

    return dset_name, train_dset, val_dset, collate_fn


def get_voc_dataset(root):
    from dataset.voc_dataset import VOCDataset, custom_collate_fn
    dset_name = 'voc2012'

    transform_og = transforms.Compose([transforms.Resize((416, 416)),
                                       transforms.ToTensor()])
    transform_norm = transforms.Compose([transforms.Resize((416, 416)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_noise = transforms.Compose([transforms.Resize((416, 416)),
                                          transforms.ToTensor(),
                                          GaussianNoise(mean=0, std=.2)])
    transform_norm_noise = transforms.Compose([transforms.Resize((416, 416)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
                                               GaussianNoise(mean=0, std=.2)])
    transform_rotate = transforms.Compose([transforms.Resize((416, 416)),
                                            transforms.RandomRotation((-60, 60)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_vflip = transforms.Compose([transforms.Resize((416, 416)),
                                                   transforms.RandomVerticalFlip(1),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_hflip = transforms.Compose([transforms.Resize((416, 416)),
                                                     transforms.RandomHorizontalFlip(1),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    dset_og = VOCDataset(root, img_size=(416, 416), transforms=transform_og, is_categorical=True)
    dset_norm = VOCDataset(root, img_size=(416, 416), transforms=transform_norm, is_categorical=True)
    dset_noise = VOCDataset(root, img_size=(416, 416), transforms=transform_noise, is_categorical=True)
    dset_norm_noise = VOCDataset(root, img_size=(416, 416), transforms=transform_norm_noise, is_categorical=True)
    # dset_rotate = VOCDataset(root, img_size=(416, 416), transforms=transform_rotate, is_categorical=True)
    # dset_vflip = VOCDataset(root, img_size=(416, 416), transforms=transform_vflip, is_categorical=True)
    # dset_hflip = VOCDataset(root, img_size=(416, 416), transforms=transform_hflip, is_categorical=True)

    num_classes = dset_og.num_classes

    n_data = len(dset_og)
    n_train_data = int(n_data * .7)
    indices = list(range(n_data))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[:n_train_data], indices[n_train_data:]
    train_dset_og = Subset(dset_og, indices=train_idx)
    train_dset_norm = Subset(dset_norm, indices=train_idx)
    train_dset_noise = Subset(dset_noise, indices=train_idx)
    train_dset_norm_noise = Subset(dset_norm_noise, indices=train_idx)
    # train_dset_rotate = Subset(dset_rotate, indices=train_idx)
    # train_dset_vflip = Subset(dset_vflip, indices=train_idx)
    # train_dset_hflip = Subset(dset_hflip, indices=train_idx)

    # train_dset = ConcatDataset([dset_og, dset_norm, dset_noise, dset_norm_noise])
    train_dset = train_dset_og
    val_dset = Subset(dset_og, indices=val_idx)

    collate_fn = custom_collate_fn

    return dset_name, train_dset, val_dset, collate_fn


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--lr', type=float, required=False, default=.0001)
    parser.add_argument('--weight_decay', type=float, required=False, default=.0005)
    parser.add_argument('--momentum', type=float, required=False, default=.9)
    parser.add_argument('--num_epochs', type=int, required=False, default=50)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    num_epochs = args.num_epochs

    model_save_term = 1

    # Generate COCO dataset
    root = 'D://DeepLearningData/COCOdataset2017/'
    dset_name, train_dset, val_dset, collate_fn = get_coco_dataset(root)
    num_classes = 92

    # Generate VOC dataset
    # root = 'D://DeepLearningData/VOC2012'
    # dset_name, train_dset, val_dset, collate_fn = get_voc_dataset(root)
    # num_classes = 20

    # Generate data loaders
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

    # Load model
    model_name = 'yolov2mobile_backbone'
    anchor_box_samples = torch.Tensor([[1.73145, 1.3221],
                                       [4.00944, 3.19275],
                                       [8.09892, 5.05587],
                                       [4.84053, 9.47112],
                                       [10.0071, 11.2364]])
    model = YOLOV2Mobile(in_size=(416, 416), num_classes=num_classes, anchor_box_samples=anchor_box_samples).to(device)
    state_dict_pth = None
    # state_dict_pth = 'pretrained models/yolov2mobile19_coco2017_16epoch_0.0001lr_20.36563loss_5.41336losscoord_10.7989938721lossconf_4.15328losscls.pth'
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth), strict=False)

    # Define optimizer, loss function
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.9)

    # Define anchor box configuration
    dummy = torch.zeros(1, 3, 416, 416).to(device)
    pred_dummy = model(dummy)

    anchor_box_base = get_output_anchor_box_tensor(anchor_box_sizes=anchor_box_samples, out_size=pred_dummy.shape[1:3]).to(device)
    # for idx1 in range(13):
    #     for idx2 in range(13):
    #         for idx3 in range(5):
    #             print(f'({idx1}, {idx2}, {idx3}) {anchor_box_base[idx1, idx2, 4 * idx3:4 * (idx3 + 1)]}')

    num_iter = 0
    train_loss_list = []
    train_loss_coord_list = []
    train_loss_confidence_list = []
    train_loss_class_list = []
    val_loss_list = []
    val_loss_coord_list = []
    val_loss_confidence_list = []
    val_loss_class_list = []
    model.train()

    t_start = time.time()
    for e in range(num_epochs):
        num_batches = 0
        num_datas = 0
        train_loss = 0
        train_loss_coord = 0
        train_loss_confidence = 0
        train_loss_class = 0

        t_train_start = time.time()
        for i, (imgs, anns) in enumerate(train_loader):
            t_batch_start = time.time()
            num_batches += 1
            num_datas += len(imgs)
            num_iter += 1
            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print(f'({num_iter}) ', end='')
            print('{}/{} '.format(num_datas, len(train_dset)), end='')

            x = make_batch(imgs).to(device)

            predict_temp = model(x)
            predict_list = []
            y_list = []

            for b in range(len(anns)):
                ground_truth_box = anns[b]['bbox']
                label_categorical = anns[b]['label_categorical']

                # print(f'{label_int} {label_categorical} {name} {fn} {ground_truth_box / 32}')

                h_img, w_img = 416, 416
                ratio_y, ratio_x = 1 / 32, 1 / 32
                ground_truth_box = torch.as_tensor(ground_truth_box)
                if len(ground_truth_box.shape) < 2:
                    ground_truth_box = ground_truth_box.unsqueeze(0)

                predict_list.append(get_yolo_v2_output_tensor(predict_temp[b], anchor_box_base))
                y_temp = get_yolo_v2_target_tensor(ground_truth_boxes=ground_truth_box,
                                                    anchor_boxes=anchor_box_base,
                                                    labels=label_categorical,
                                                    n_bbox_predict=5,
                                                    n_class=num_classes,
                                                    in_size=(h_img, w_img),
                                                    out_size=(13, 13))
                y_list.append(y_temp)

            y = make_batch(y_list).to(device)
            predict = make_batch(predict_list).to(device)

            # for idx1 in range(13):
            #     for idx2 in range(13):
            #         for idx3 in range(5):
            #             if y[0, idx1, idx2, 25 * idx3 + 4] != 0:
            #                 print('\nPredict_temp: ({}, {}, {}) {}'.format(idx1, idx2, idx3, predict_temp[0, idx1, idx2, 25 * idx3:25 * (idx3 + 1)]))
            #                 print('\nPredict: ({}, {}, {}) {}'.format(idx1, idx2, idx3, predict[0, idx1, idx2, 25 * idx3:25 * (idx3 + 1)]))
            #                 print('y: ({}, {}, {}) {}'.format(idx1, idx2, idx3, y[0, idx1, idx2, 25 * idx3:25 * (idx3 + 1)]))

            del x, predict_temp, predict_list, y_list, y_temp

            optimizer.zero_grad()
            loss, loss_coord, loss_confidence, loss_class = loss_func(predict=predict, target=y, anchor_boxes=anchor_box_base,
                                                                      num_bbox_predict=5, num_classes=num_classes, lambda_coord=1)

            if loss.detach().cpu().item() > 1000:
                for ann in anns:
                    print(ann['file_name'])

            if torch.isnan(loss).sum() > 0:
                print('NaN appears.')
                exit(0)

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item()
            train_loss_coord += loss_coord.detach().cpu().item()
            train_loss_confidence += loss_confidence.detach().cpu().item()
            train_loss_class += loss_class.detach().cpu().item()

            t_batch_end = time.time()

            H, M, S = time_calculator(t_batch_end - t_start)

            print('<loss> {: <8.5f}  <loss_coord> {: <8.5f}  <loss_confidence> {: <8.5f}  <loss_class> {: <8.5f}  '.format(
                loss.detach().cpu().item(), loss_coord.detach().cpu().item(),
                loss_confidence.detach().cpu().item(), loss_class.detach().cpu().item()), end='')
            print('<loss_avg> {: <8.5f}  <loss_coord_avg> {: <8.5f}  <loss_confidence_avg> {: <8.5f}  <loss_class_avg> {: <8.5f}  '.format(
                train_loss / num_batches, train_loss_coord / num_batches, train_loss_confidence / num_batches,
                train_loss_class / num_batches
            ), end='')
            print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

            if num_iter > 0 and num_iter % 5000 == 0:
                save_pth = 'saved models/ckp_{}_{}_{}epoch_{}iter_{}lr.pth'.format(
                    model_name, dset_name, e + 1, num_iter, learning_rate)
                torch.save(model.state_dict(), save_pth)

            del y, predict, loss

        train_loss /= num_batches
        train_loss_coord /= num_batches
        train_loss_confidence /= num_batches
        train_loss_class /= num_batches

        train_loss_list.append(train_loss)
        train_loss_coord_list.append(train_loss_coord)
        train_loss_confidence_list.append(train_loss_confidence)
        train_loss_class_list.append(train_loss_class)

        t_train_end = time.time()
        H, M, S = time_calculator(t_train_end - t_train_start)

        print('        <train_loss> {: <8.5f}  <train_loss_coord> {: <8.5f}  <train_loss_confidence> {: <8.5f}  <train_loss_class> {: <8.5f}  '.format(
            train_loss_list[-1], train_loss_coord_list[-1], train_loss_confidence_list[-1], train_loss_class_list[-1]), end='')
        print('<time> {:02d}:{:02d}:{:02d} '.format(int(H), int(M), int(S)))

        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_loss_coord = 0
            val_loss_confidence = 0
            val_loss_class = 0
            num_batches = 0

            for i, (imgs, anns) in enumerate(val_loader):
                num_batches += 1

                x = make_batch(imgs).to(device)

                predict_temp = model(x)
                predict_list = []
                y_list = []

                for b in range(len(anns)):
                    h_img, w_img = 416, 416
                    ground_truth_box = anns[b]['bbox']
                    label = anns[b]['label_categorical']

                    ratio_h, ratio_w = 1 / 32, 1 / 32
                    ground_truth_box = torch.as_tensor(ground_truth_box)
                    if len(ground_truth_box.shape) < 2:
                        ground_truth_box = ground_truth_box.unsqueeze(0)
                    # if len(ground_truth_box) > 0:
                    #     ground_truth_box[:, 0] *= ratio_h
                    #     ground_truth_box[:, 1] *= ratio_w
                    #     ground_truth_box[:, 2] *= ratio_h
                    #     ground_truth_box[:, 3] *= ratio_w

                    predict_list.append(get_yolo_v2_output_tensor(predict_temp[b], anchor_box_base))
                    y_list.append(get_yolo_v2_target_tensor(ground_truth_boxes=ground_truth_box,
                                                            labels=label,
                                                            anchor_boxes=anchor_box_base,
                                                            n_bbox_predict=5,
                                                            n_class=num_classes,
                                                            in_size=(h_img, w_img),
                                                            out_size=(13, 13)))

                y = make_batch(y_list).to(device)
                predict = make_batch(predict_list).to(device)

                del predict_temp, predict_list, y_list

                loss, loss_coord, loss_confidence, loss_class = loss_func(predict=predict, target=y, anchor_boxes=anchor_box_base,
                                                                          num_bbox_predict=5, num_classes=num_classes, lambda_coord=1)

                val_loss += loss.detach().cpu().item()
                val_loss_coord += loss_coord.detach().cpu().item()
                val_loss_confidence += loss_confidence.detach().cpu().item()
                val_loss_class += loss_class.detach().cpu().item()

                del x, y, predict, loss

            val_loss /= num_batches
            val_loss_coord /= num_batches
            val_loss_confidence /= num_batches
            val_loss_class /= num_batches

            val_loss_list.append(val_loss)
            val_loss_coord_list.append(val_loss_coord)
            val_loss_confidence_list.append(val_loss_confidence)
            val_loss_class_list.append(val_loss_class)

            print('        <val_loss> {: <10.5f}  <val_loss_coord> {: <10.5f}  <val_loss_confidence> {: <10.10f}  <val_loss_class> {: <10.5f}'.format(
                val_loss_list[-1], val_loss_coord_list[-1], val_loss_confidence_list[-1], val_loss_class_list[-1]))

            if (e + 1) % model_save_term == 0:
                save_pth = 'saved models/{}_{}_{}epoch_{}lr_{:.5f}loss_{:.5f}losscoord_{:.10f}lossconf_{:.5f}losscls.pth'.format(
                    model_name, dset_name, e + 1, learning_rate, val_loss_list[-1], val_loss_coord_list[-1],
                    val_loss_confidence_list[-1], val_loss_class_list[-1])
                torch.save(model.state_dict(), save_pth)

    x_axis = [i for i in range(len(train_loss_list))]

    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b-', label='Validation')
    plt.title('Train/Validation loss')
    plt.legend()

    plt.show()
























































