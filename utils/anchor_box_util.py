import numpy as np
import torch

from utils.pytorch_util import calculate_iou


def distance_metric(bbox1, bbox2):
    return 1 - calculate_iou(bbox1, bbox2)


def k_means_cluster_anchor_box(bbox, K):
    # Inputs:
    #    bbox: list [tensor (y1, x1, y2, x2)]
    #    K: int # centroid
    # Outputs:
    #    centroids: tensor [K, (y1, x1, y2, x2)]
    #    clusters: ndarray [# bounding box]

    print('Start K-means clustering!')
    centroids = np.array([bbox[0]])
    clusters = np.array([-1 for _ in range(len(bbox))])
    changed = True
    max_iter = 50

    print('Selecting initial centroids...')
    for i in range(1, K):
        max_dist = -1
        arg_max_dist = -1
        for b, box in enumerate(bbox):
            if box in centroids:
                continue
            dist = 0
            for cent in centroids:
                dist += distance_metric(box, cent)
            if dist > max_dist:
                max_dist = dist
                arg_max_dist = b
        centroids = np.concatenate([centroids, [bbox[arg_max_dist]]], axis=0)

    print('Clustering...')
    iter_ = 0
    while changed and iter_ < max_iter:
        print(iter_, clusters[:20])
        iter_ += 1
        changed = False
        for b, box in enumerate(bbox):
            min_dist = -1
            arg_min_dist = -1
            for c, cent in enumerate(centroids):
                dist = calculate_iou(box, cent)
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    arg_min_dist = c
            if clusters[b] != arg_min_dist:
                clusters[b] = arg_min_dist
                changed = True

        if changed:
            for i in range(K):
                args = np.where(clusters == i)
                bboxes_cluster = bbox[args]
                if len(bboxes_cluster) == 0:
                    continue
                x1, y1, x2, y2 = np.mean(bboxes_cluster, axis=0)
                new_cent = np.array(torch.Tensor([y1, x1, y2, x2]))
                centroids[i] = new_cent

    centroids = torch.as_tensor(centroids)

    print('K-means clustering done!')

    return centroids, clusters