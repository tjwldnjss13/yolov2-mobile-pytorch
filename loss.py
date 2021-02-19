import torch

from utils.pytorch_util import calculate_iou
from utils.anchor_box_util import convert_box_from_hw_to_yx

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def yolo_pretrain_custom_loss(predict, target):
    losses = -1 * (target * torch.log(predict + 1e-15) + (1 - target) * torch.log(1 - predict + 1e-15))
    batch = losses.shape[0]
    loss = losses.sum() / batch

    return loss


def yolo_custom_loss(predict, target, num_bbox_predict, num_classes, lambda_coord=5, lambda_noobj=.5):
    """
    :param predict: tensor, [batch, height, width, (cy, cx, h, w, p) * num bounding boxes]
    :param target: tensor, [batch, height, width, (cy, cx, h, w, p)]
    :param num_bbox_predict: int
    :param num_classes: int
    :param lambda_coord: float
    :param lambda_noobj: float

    :return: float tensor, [float]
    """
    NUM_BATCH, H, W = predict.shape[:3]

    pred = predict.reshape(NUM_BATCH, -1, 5 + num_classes)
    tar = target.reshape(NUM_BATCH, -1, 5 + num_classes)

    # obj_responsible_mask = torch.zeros(NUM_BATCH, H * W, 5).to(device)

    # for i in range(num_bbox_predict):
    #     obj_responsible_mask[:, :, :, i] = target[:, :, :, 4]

    # Get responsible masks
    pred_bboxes = pred[:, :, :4]
    pred_probs = pred[:, :, 4]

    tar_bboxes = tar[:, :, :4]
    tar_probs = tar[:, :, 4]  # [num batch, h * w * 5(num predict bbox)]

    pred_y1 = pred_bboxes[:, :, 0] - .5 * pred_bboxes[:, :, 2]
    pred_x1 = pred_bboxes[:, :, 1] - .5 * pred_bboxes[:, :, 3]
    pred_y2 = pred_bboxes[:, :, 0] + pred_bboxes[:, :, 2]
    pred_x2 = pred_bboxes[:, :, 1] + pred_bboxes[:, :, 3]

    pred_bboxes = torch.cat([pred_y1, pred_x1, pred_y2, pred_x2], dim=2)

    tar_y1 = tar_bboxes[:, :, 0] - .5 * tar_bboxes[:, :, 2]
    tar_x1 = tar_bboxes[:, :, 1] - .5 * tar_bboxes[:, :, 3]
    tar_y2 = tar_bboxes[:, :, 0] + tar_bboxes[:, :, 2]
    tar_x2 = tar_bboxes[:, :, 1] + tar_bboxes[:, :, 3]

    tar_bboxes = torch.cat([tar_y1, tar_x1, tar_y2, tar_x2], dim=2)

    indices_valid = torch.where(tar_probs == 1)
    pred_bboxes_valid = pred_bboxes[indices_valid]
    tar_bboxes_valid = tar_bboxes[indices_valid]

    ious_valid = calculate_iou(pred_bboxes_valid, tar_bboxes_valid, dim=2)

    ious = torch.zeros(NUM_BATCH, H * W * 5).to(device)  # [num batch, h * w * 5(num predict bbox)]
    ious[indices_valid] = ious_valid
    ious = ious.reshape(NUM_BATCH, H * W, 5)

    indices_argmax_ious = torch.argmax(ious, dim=2)

    obj_responsible_mask = torch.zeros(NUM_BATCH, H * W, 5).to(device)  # [num batch, h * w, 5(num predict bbox)]
    obj_responsible_mask[indices_argmax_ious] = 1

    no_obj_responsible_mask = torch.zeros(NUM_BATCH, H * W, 5).to(device)
    no_obj_responsible_mask[indices_argmax_ious[:-1]] = 1
    no_obj_responsible_mask[indices_argmax_ious] = 0

    responsible_mask = torch.zeros(obj_responsible_mask.shape[:-1]).to(device)
    for i in range(num_bbox_predict):
        responsible_mask += obj_responsible_mask[:, :, i]

    # Get coordinate loss(1)
    loss_coord = torch.square(pred[:, :, 0] - tar[:, :, 0]) + \
                 torch.square(pred[:, :, 1] - tar[:, :, 1]) + \
                 torch.square(torch.sqrt(pred[:, :, 2]) - torch.sqrt(tar[:, :, 2])) + \
                 torch.square(torch.sqrt(pred[:, :, 3]) - torch.sqrt(tar[:, :, 3]))
    loss_coord *= lambda_coord * obj_responsible_mask

    # Get confidence loss(2)
    loss_confidence = torch.square(pred[:, :, 4] - tar[:, :, 4]) + \
                      lambda_noobj * no_obj_responsible_mask * torch.square(pred[:, :, 4] - tar[:, :, 4])

    # Get class loss(3)
    loss_class = torch.square(pred[:, :, 5:] - tar[:, :, 5:])
    loss_class = loss_class.reshape(NUM_BATCH, H * W, -1)
    loss_class = torch.sum(loss_class, dim=2)
    loss_class *= responsible_mask

    # Sum up all the losses
    loss = loss_coord + loss_confidence + loss_class

    return loss / NUM_BATCH


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

    class_losses = torch.zeros(h, w).to(device)

    coord_loss = torch.zeros(1).to(device)
    confidence_loss = torch.zeros(1).to(device)
    class_loss = torch.zeros(1).to(device)

    n_batch = predict.shape[0]
    for b in range(n_batch):
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

            # Confidence loss
            confidence_losses_temp = torch.square(predict[b, :, :, (5 + n_class) * i + 4] - target[b, :, :, (5 + n_class) * i + 4])
            confidence_loss_batch += (confidence_losses_temp * obj_responsible_mask[:, :, i] \
                                     + lambda_noobj * confidence_losses_temp * no_obj_responsible_mask[:, :, i]).sum()

            # Class loss
            class_losses_temp = torch.square(predict[b, :, :, (5 + n_class) * i + 5:(5 + n_class) * (i + 1)] -
                                             target[b, :, :, (5 + n_class) * i + 5:(5 + n_class) * (i + 1)]).sum(dim=2)
            class_loss_batch += (responsible_mask * class_losses_temp).sum()

        coord_loss += coord_loss_batch
        confidence_loss += confidence_loss_batch
        class_loss += class_loss_batch

    coord_loss = lambda_coord * coord_loss / n_batch
    confidence_loss = confidence_loss / n_batch
    class_loss = class_losses.sum() / n_batch

    loss = coord_loss + confidence_loss + class_loss
    print(coord_loss.detach().cpu().item(), confidence_loss.detach().cpu().item(), class_loss.detach().cpu().item())

    return loss

