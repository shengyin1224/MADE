import torch
import numpy as np 
from data.obj_util import *

def get_pred_box(pred_selected):
    """
    Single sample input
        pred_selected: List[Dict["pred", "score", "selected_idx", "is_nan"], ...]
            "pred": (n, 1, 4, 2)
            "score": (n, )
    output:
        [cls1_det, cls2_det, ...]
        per-class detected bboxes.
    """
    per_class_boxes = []
    for p in pred_selected:
        cls_pred_corners = p['pred'][:, 0]
        cls_pred_scores = p['score']
        pred_corners = cls_pred_corners.reshape(cls_pred_corners.shape[0], np.prod(cls_pred_corners.shape[1:]))  # avoid shape[0] == 0
        pred_corners = np.hstack((pred_corners, cls_pred_scores[:, np.newaxis]))  # (n, 9)
        per_class_boxes.append(pred_corners)
    return per_class_boxes  # bboxes of one sample


def get_gt_box(anchors_map, reg_targets, gt_max_iou_idx):
    """
    Single sample input, do not support batch
        anchors_map: (256, 256, 6, 6)
        reg_targets: (256, 256, 6, 1, 6)
        gt_max_iou_idx: (n, 4)
    
    output:
        list of box corners in a sample
    """
    gt_corners = []
    for k in range(len(gt_max_iou_idx)):
        anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]
        encode_box = reg_targets[tuple(gt_max_iou_idx[k][:-1])+(0,)]

        decode_box = bev_box_decode_np(encode_box, anchor)  # do not support batch
        decode_corner = center_to_corner_box2d(decode_box[np.newaxis, :2], 
                                                decode_box[np.newaxis, 2:4], 
                                                decode_box[np.newaxis, 4:])[0]  # batch operation
        gt_corners.append(decode_corner)
    gt_corners = np.array(gt_corners) # (n, 4, 2)
    gt_corners = gt_corners.reshape(gt_corners.shape[0], -1) # (n, 8)

    annotations_frame = {"bboxes": gt_corners,
                        "labels": np.zeros(gt_corners.shape[0])}
    return annotations_frame
