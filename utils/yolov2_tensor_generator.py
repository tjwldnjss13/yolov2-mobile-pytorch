import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_output_anchor_box_tensor(anchor_box_sizes, out_size):
    """
    Make anchor box the same shape as output's.
    :param anchor_box_sizes: tensor, [# anchor box, (height, width)]
    :param out_size: tuple or list, (height, width)

    :return tensor, [height, width, (cy, cx, h, w) * num bounding box]
    """

    out = torch.zeros(out_size[0], out_size[1], 4 * len(anchor_box_sizes)).to(device)
    cy_ones = torch.ones(1, out_size[1])
    cx_ones = torch.ones(out_size[0], 1)
    cy_tensor = torch.zeros(1, out_size[1])
    cx_tensor = torch.zeros(out_size[0], 1)
    for i in range(1, out_size[0]):
        cy_tensor = torch.cat([cy_tensor, cy_ones * i], dim=0)
        cx_tensor = torch.cat([cx_tensor, cx_ones * i], dim=1)

    ctr_tensor = torch.cat([cy_tensor.unsqueeze(2), cx_tensor.unsqueeze(2)], dim=2)

    for i in range(len(anchor_box_sizes)):
        out[:, :, 4 * i:4 * i + 2] = ctr_tensor
        out[:, :, 4 * i + 2] = anchor_box_sizes[i, 0]
        out[:, :, 4 * i + 3] = anchor_box_sizes[i, 1]

    return out


# anc_sizes = torch.Tensor([[12, 23], [24, 15]])
# outs = get_output_anchor_box_tensor(anc_sizes, (13, 13))
# print(outs)


def get_yolo_v2_output_tensor(deltas, anchor_boxes):
    """
    :param deltas: tensor, [height, width, ((dy, dx, dh, dw, p) + class scores) * num anchor boxes]
    :param anchor_boxes: tensor, [height, width, (cx, cy, h, w) * num anchor boxes]

    :return: tensor, [height, width, ((cy, cx, h, w, p) + class scores) * num anchor boxes]
    """

    out = torch.zeros(deltas.shape).to(device)
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=2)

    num_anchor_boxes = int(anchor_boxes.shape[2] / 4)
    num_data_per_box = int(deltas.shape[2] / num_anchor_boxes)

    for i in range(num_anchor_boxes):
        out[:, :, num_data_per_box * i: num_data_per_box * i + 2] = \
            sigmoid(deltas[:, :, num_data_per_box * i:num_data_per_box * i + 2]) + \
            anchor_boxes[:, :, 4 * i:4 * i + 2]
        out[:, :, 4 * i:4 * i + 2] = torch.exp(deltas[:, :, num_data_per_box * i + 2:num_data_per_box * i + 4]) * \
                                     anchor_boxes[:, :, 4 * i + 2:4 * (i + 1)]
        out[:, :, num_data_per_box * i + 4] = sigmoid(deltas[:, :, num_data_per_box * i + 4])
        out[:, :, num_data_per_box * i + 5:num_data_per_box * (i + 1)] = \
            softmax(deltas[:, :, num_data_per_box * i + 5:num_data_per_box * (i + 1)])

    return out


def get_yolo_v2_target_tensor(ground_truth_boxes, labels, n_bbox_predict, n_class, in_size, out_size):
    """
    :param ground_truth_boxes: tensor, [num ground truth, (y1, x1, y2, x2)]
    :param labels: tensor, [num bounding boxes, (p0, p1, ..., pn)]
    :param n_bbox_predict: int
    :param n_class: int
    :param in_size: tuple or list, (height, width)
    :param out_size: tuple or list, (height, width)

    :return: tensor, [height of output, width of output, (cy, cx, h, w, p) * num bounding boxes]
    """

    bboxes = ground_truth_boxes

    n_gt = len(bboxes)
    in_h, in_w = in_size
    out_h, out_w = out_size

    ratio = out_h / in_h

    target = torch.zeros((out_h, out_w, (5 + n_class) * n_bbox_predict))

    for i in range(n_gt):
        bbox = bboxes[i]
        h, w = (bbox[2] - bbox[0]) / in_h, (bbox[3] - bbox[1]) / in_w  # Height, width is relative to original image
        y, x = (bbox[0] + .5 * h) * ratio, (bbox[1] + .5 * w) * ratio

        y_cell_idx, x_cell_idx = int(y), int(x)
        y_cell, x_cell = y - int(y), x - int(x)
        label = labels[i]

        for j in range(n_bbox_predict):
            target[y_cell_idx, x_cell_idx, (5 + n_class) * j] = x_cell
            target[y_cell_idx, x_cell_idx, (5 + n_class) * j + 1] = y_cell
            target[y_cell_idx, x_cell_idx, (5 + n_class) * j + 2] = w
            target[y_cell_idx, x_cell_idx, (5 + n_class) * j + 3] = h
            target[y_cell_idx, x_cell_idx, (5 + n_class) * j + 4] = 1

            target[y_cell_idx, x_cell_idx, (5 + n_class) * j + 5:(5 + n_class) * (j + 1)] = label

    return target
