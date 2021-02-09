import os
import time
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from utils.util import time_calculator
from utils.pytorch_util import make_batch
from utils.yolov2_tensor_generator import get_output_anchor_box_tensor, get_yolo_v2_output_tensor, get_yolo_v2_target_tensor
from dataset.coco_dataset import COCODataset, custom_collate_fn
from models.yolov2_model import YOLOV2Mobile
from loss import yolo_custom_loss


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--lr', type=float, required=False, default=.005)
    parser.add_argument('--weight_decay', type=float, required=False, default=.005)
    parser.add_argument('--num_epochs', type=int, required=False, default=20)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs

    # Generate COCO dataset
    dset_name = 'coco2017'
    root = 'C://DeepLearningData/COCOdataset2017/'
    train_img_dir = os.path.join(root, 'images', 'train')
    val_img_dir = os.path.join(root, 'images', 'val')
    train_ann_pth = os.path.join(root, 'annotations', 'instances_train2017.json')
    val_ann_pth = os.path.join(root, 'annotations', 'instances_val2017.json')
    transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])

    train_dset = COCODataset(root=root, images_dir=train_img_dir, annotation_path=train_ann_pth, is_categorical=True, transforms=transform)
    val_dset = COCODataset(root=root, images_dir=val_img_dir, annotation_path=val_ann_pth, is_categorical=True, transforms=transform)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    num_classes = train_dset.num_classes

    # Load model
    model_name = 'yolov2mobile'
    anchor_box_samples = torch.Tensor([[86, 161], [95, 153], [169, 183], [203, 180], [45, 48]])
    model = YOLOV2Mobile(in_size=(416, 416), num_classes=num_classes, anchor_box_samples=anchor_box_samples).to(device)
    state_dict_pth = None
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth))

    # Define optimizer, loss function
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = yolo_custom_loss

    # Define anchor box configuration
    dummy = torch.zeros(1, 3, 416, 416).to(device)
    pred_dummy = model(dummy)

    anchor_box_sizes = torch.Tensor([[86, 161], [95, 153], [169, 183], [203, 180], [45, 48]])
    anchor_box_base = get_output_anchor_box_tensor(anchor_box_sizes=anchor_box_samples, out_size=pred_dummy.shape[1:3]).to(device)

    train_loss_list = []
    val_loss_list = []
    model.train()

    t_start = time.time()
    for e in range(num_epochs):
        num_batches = 0
        num_datas = 0
        train_loss = 0

        t_train_start = time.time()
        for i, (imgs, anns) in enumerate(train_loader):
            t_batch_start = time.time()
            num_batches += 1
            num_datas += len(imgs)
            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print('{}/{} '.format(num_datas, len(train_dset)), end='')

            x = make_batch(imgs).to(device)

            predict_temp = model(x)
            predict_list = []
            y_list = []

            for b in range(len(anns)):
                h_img, w_img = anns[b]['height'], anns[b]['width']
                ground_truth_box = anns[b]['bbox']
                label = anns[b]['label']

                ratio_h, ratio_w = 416 / h_img, 416 / w_img
                ground_truth_box = torch.as_tensor(ground_truth_box)
                if len(ground_truth_box.shape) < 2:
                    ground_truth_box.unsqueeze(0)
                if len(ground_truth_box) > 0:
                    ground_truth_box[:, 0] *= ratio_h
                    ground_truth_box[:, 1] *= ratio_w
                    ground_truth_box[:, 2] *= ratio_h
                    ground_truth_box[:, 3] *= ratio_w

                predict_list.append(get_yolo_v2_output_tensor(predict_temp[b], anchor_box_base))
                y_list.append(get_yolo_v2_target_tensor(ground_truth_boxes=ground_truth_box,
                                                        labels=label,
                                                        n_bbox_predict=5,
                                                        n_class=num_classes,
                                                        in_size=(416, 416),
                                                        out_size=(13, 13)))

            y = make_batch(y_list).to(device)
            predict = make_batch(predict_list).to(device)

            del predict_temp, predict_list, y_list

            optimizer.zero_grad()
            loss = loss_func(predict=predict, target=y, n_bbox_predict=5, n_class=num_classes)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item()

            t_batch_end = time.time()

            H, M, S = time_calculator(t_batch_end - t_start)

            print('<loss> {:<21} <loss_avg> {:<21} '.format(loss.detach().cpu().item(), train_loss / num_batches), end='')
            print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

            del x, y, predict, loss

        train_loss /= num_batches
        train_loss_list.append(train_loss)

        t_train_end = time.time()
        H, M, S = time_calculator(t_train_end - t_train_start)

        print('        <train_loss> {:<21} '.format(train_loss_list[-1]), end='')
        print('<time> {:02d}:{:02d}:{:02d} '.format(int(H), int(M), int(S)), end='')

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for i, (imgs, anns) in enumerate(val_loader):
                num_batches += 1

                x = make_batch(imgs).to(device)

                predict_temp = model(x)
                predict_list = []
                y_list = []

                for b in range(len(anns)):
                    h_img, w_img = anns[b]['height'], anns[b]['width']
                    ground_truth_box = anns[b]['bbox']
                    label = anns[b]['label']

                    ratio_h, ratio_w = 416 / h_img, 416 / w_img
                    ground_truth_box = torch.as_tensor(ground_truth_box)
                    if len(ground_truth_box.shape) < 2:
                        ground_truth_box.unsqueeze(0)
                    if len(ground_truth_box) > 0:
                        ground_truth_box[:, 0] *= ratio_h
                        ground_truth_box[:, 1] *= ratio_w
                        ground_truth_box[:, 2] *= ratio_h
                        ground_truth_box[:, 3] *= ratio_w

                    predict_list.append(get_yolo_v2_output_tensor(predict_temp, anchor_box_base))
                    y_list.append(get_yolo_v2_target_tensor(ground_truth_boxes=ground_truth_box,
                                                            labels=label,
                                                            n_bbox_predict=5,
                                                            n_class=num_classes,
                                                            in_size=(416, 416),
                                                            out_size=(13, 13)))

                y = make_batch(y_list).to(device)
                predict = make_batch(predict_list).to(device)

                del predict_temp, predict_list, y_list

                loss = loss_func(predict=predict, target=y, n_bbox_predict=5, n_class=num_classes)

                val_loss += loss.detach().cpu().item()

                del x, y, predict, loss

            val_loss /= num_batches
            val_loss_list.append(val_loss)

            print('<val_loss> {:<21}'.format(val_loss_list[-1]))

            if (e + 1) % 2 == 0:
                save_pth = 'saved models/{}_{}_{}epoch_{}lr_{:.5f}loss.pth'.format(model_name, dset_name, e + 1, learning_rate, val_loss_list[-1])
                torch.save(model.state_dict(), save_pth)

    x_axis = [i for i in range(num_epochs)]

    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b-', label='Validation')
    plt.title('Train/Validation loss')
    plt.legend()

    plt.show()
























































