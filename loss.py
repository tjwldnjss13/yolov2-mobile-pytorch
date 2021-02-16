import torch

from utils.pytorch_util import calculate_iou
from utils.anchor_box_util import convert_box_from_hw_to_yx

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def yolo_pretrain_custom_loss(predict, target):
    losses = -1 * (target * torch.log(predict + 1e-15) + (1 - target) * torch.log(1 - predict + 1e-15))
    batch = losses.shape[0]
    loss = losses.sum() / batch

    return loss


def yolo_custom_loss(predict, target, n_bbox_predict, n_class, lambda_coord=5, lambda_noobj=.5):
    """
    :param predict: tensor, [batch, height, width, (cy, cx, h, w, p) * num bounding boxes]
    :param target: tensor, [batch, height, width, (cy, cx, h, w, p) * num bounding boxes]
    :param n_bbox_predict: int
    :param n_class: int
    :param lambda_coord: float
    :param lambda_noobj: float

    :return: float tensor, [float]
    """

    h, w = predict.shape[1:3]

    coord_losses = torch.zeros(h, w, 2).to(device)
    confidence_losses = torch.zeros(h, w).to(device)
    class_losses = torch.zeros(h, w).to(device)

    coord_loss = torch.zeros(1).to(device)
    confidence_loss = torch.zeros(1).to(device)
    class_loss = torch.zeros(1).to(device)

    n_batch = predict.shape[0]
    for b in range(n_batch):
        is_obj_global = torch.zeros(h, w).to(device)

        obj_responsible_mask = torch.zeros(h, w, n_bbox_predict).to(device)
        no_obj_responsible_mask = torch.zeros(h, w, n_bbox_predict).to(device)

        # Get responsible box masks
        for i in range(n_bbox_predict):
            obj_responsible_mask[:, :, i] = target[b, :, :, (5 + n_class) * i + 4]
            no_obj_responsible_mask[:, :, i] = target[b, :, :, (5 + n_class) * i + 4]

        for s1 in range(7):
            for s2 in range(7):
                if obj_responsible_mask[s1, s2, 0] == 1:
                    ious = torch.zeros(n_bbox_predict)

                    for n in range(n_bbox_predict):
                        box_temp = convert_box_from_hw_to_yx(predict[b, s1, s2, (5 + n_class) * n:(5 + n_class) * n + 4]).to(device)
                        gt = target[b, s1, s2, :4]
                        ious[n] = calculate_iou(box_temp, gt)

                    idx_max_iou = ious.argmax().item()

                    for n in range(n_bbox_predict):
                        if n != idx_max_iou:
                            obj_responsible_mask[s1, s2, n] = 0
                        else:
                            no_obj_responsible_mask[s1, s2, n] = 0

        responsible_mask = torch.zeros(h, w).to(device)
        for n in range(n_bbox_predict):
            responsible_mask += obj_responsible_mask[:, :, n]

        # Calculate losses
        coord_loss_batch = torch.zeros(1).to(device)
        confidence_loss_batch = torch.zeros(1).to(device)
        class_loss_batch = torch.zeros(1).to(device)

        for i in range(n_bbox_predict):
            # Coordinate loss
            coord_losses_temp = torch.square(predict[b, :, :, (5 + n_class) * i] - target[b, :, :, (5 + n_class) * i]) \
                                + torch.square(predict[b, :, :, (5 + n_class) * i + 1] - target[b, :, :, (5 + n_class) * i + 1]) \
                                + torch.square(torch.sqrt(predict[b, :, :, (5 + n_class) * i + 2]) - torch.sqrt(target[b, :, :, (5 + n_class) * i + 2])) \
                                + torch.square(torch.sqrt(predict[b, :, :, (5 + n_class) * i + 3]) - torch.sqrt(target[b, :, :, (5 + n_class) * i + 3]))
            coord_losses_temp *= obj_responsible_mask[:, :, i]
            coord_loss_batch += coord_losses_temp.sum()

            # coord_losses += torch.square(predict[b, :, :, 5 * i] - target[b, :, :, 5 * i])
            # coord_losses += torch.square(predict[b, :, :, 5 * i + 1] - target[b, :, :, 5 * i + 1])
            # coord_losses += torch.square(torch.sqrt(predict[b, :, :, 5 * i + 2]) - torch.sqrt(target[b, :, :, 5 * i + 2]))
            # coord_losses += torch.square(torch.sqrt(predict[b, :, :, 5 * i + 3]) - torch.sqrt(target[b, :, :, 5 * i + 3]))

            # Confidence loss
            confidence_losses_temp = torch.square(predict[b, :, :, (5 + n_class) * i + 4] - target[b, :, :, (5 + n_class) * i + 4])
            confidence_loss_batch += (confidence_losses_temp * obj_responsible_mask[:, :, i] \
                                     + lambda_noobj * confidence_losses_temp * no_obj_responsible_mask[:, :, i]).sum()

            # is_obj = target[b, :, :, 5 * i + 4]
            # if b == 0:
            #     is_obj_global += is_obj
            # obj_losses_temp = torch.square(predict[b, :, :, 5 * i + 4] - is_obj)
            # obj_losses += (is_obj + (1 - is_obj) * lambda_noobj) * obj_losses_temp

            # Class loss
            class_losses_temp = torch.square(predict[b, :, :, (5 + n_class) * i + 5:(5 + n_class) * (i + 1)] -
                                             target[b, :, :, (5 + n_class) * i + 5:(5 + n_class) * (i + 1)]).sum(dim=2)
            class_loss_batch += (responsible_mask * class_losses_temp).sum()

        # if b == 0:
        #     is_obj_global /= n_batch

        coord_loss += coord_loss_batch
        confidence_loss += confidence_loss_batch
        class_loss += class_loss_batch

        # class_losses_temp = torch.square(predict[b, :, :, 5 * n_bbox_predict:] - target[b, :, :, 5 * n_bbox_predict:])
        # class_losses_temp = class_losses_temp.sum(dim=2)
        # class_losses += class_losses_temp * is_obj_global

    coord_loss = lambda_coord * coord_loss / n_batch
    confidence_loss = confidence_loss / n_batch
    # obj_loss = obj_losses.sum() / n_batch
    class_loss = class_losses.sum() / n_batch

    loss = coord_loss + confidence_loss + class_loss

    # print(coord_loss.item(), obj_loss.item(), class_loss.item(), end=' ')

    return loss

