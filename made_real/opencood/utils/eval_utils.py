# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def find_box_out(det_boxes, det_score, gt_boxes, iou_thresh):
    """
    Find gt bboxes which are out of the area which ego-agent can detect

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    iou_thresh : float
        The iou thresh.
    """
    box_out_index = []

    if gt_boxes == None:
        return box_out_index
        
    gt = gt_boxes.shape[0]
    if det_boxes is not None and det_score is not None and gt_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(len(gt_polygon_list)):
            gt_polygon = gt_polygon_list[i]
            ious = common_utils.compute_iou(gt_polygon, det_polygon_list)

            if len(det_polygon_list) == 0 or np.max(ious) < iou_thresh:
                box_out_index.append(i)
                continue

    return box_out_index

def find_box_erase_fail(det_boxes, det_score, gt_boxes, iou_thresh, erase_index):
    """
    Find gt bboxes which are out of the area which ego-agent can detect

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    iou_thresh : float
        The iou thresh.
    """
    box_out_index = []   

    gt = gt_boxes.shape[0]
    if det_boxes is not None and det_score is not None and gt_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(len(gt_polygon_list)):
            gt_polygon = gt_polygon_list[i]
            ious = common_utils.compute_iou(gt_polygon, det_polygon_list)

            if np.max(ious) >= iou_thresh:
                box_out_index.append(erase_index[i])
                continue

    return box_out_index


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, tp_list, fp_list, num_agent = 0):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()
    if num_agent == 2:
        tp_list[iou_thresh].append(len(tp))
        fp_list[iou_thresh].append(len(fp))

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt

def caluclate_tp_fp_multiclass(det_boxes_all, det_score_all, gt_boxes_all, result_stat_all, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """

    class_list = [0,1,3]
    for c in range(3):
        det_boxes = det_boxes_all[c]
        det_score = det_score_all[c]
        gt_boxes = gt_boxes_all[c]
        result_stat = result_stat_all[class_list[c]]

        if gt_boxes is None:
            continue

        # fp, tp and gt in the current frame
        fp = []
        tp = []
        gt = gt_boxes.shape[0]
        if det_boxes is not None:
            # convert bounding boxes to numpy array
            det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
            det_score = common_utils.torch_tensor_to_numpy(det_score)
            gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

            # sort the prediction bounding box by score
            score_order_descend = np.argsort(-det_score)
            det_score = det_score[score_order_descend] # from high to low
            det_polygon_list = list(common_utils.convert_format(det_boxes))
            gt_polygon_list = list(common_utils.convert_format(gt_boxes))

            # match prediction and gt bounding box, in confidence descending order
            for i in range(score_order_descend.shape[0]):
                det_polygon = det_polygon_list[score_order_descend[i]]
                ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

                if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                    fp.append(1)
                    tp.append(0)
                    continue

                fp.append(0)
                tp.append(1)

                gt_index = np.argmax(ious)
                gt_polygon_list.pop(gt_index)
            result_stat[iou_thresh]['score'] += det_score.tolist()
        result_stat[iou_thresh]['fp'] += fp
        result_stat[iou_thresh]['tp'] += tp
        result_stat[iou_thresh]['gt'] += gt

def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = np.array(iou_5['fp'])
    tp = np.array(iou_5['tp'])
    score = np.array(iou_5['score'])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results_multiclass(result_stat_dict, save_path, infer_info=None):

    for tpe in result_stat_dict.keys():
        result_stat = result_stat_dict[tpe]

        dump_dict = {}

        ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
        ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
        ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

        dump_dict.update({'ap30': ap_30,
                        'ap_50': ap_50,
                        'ap_70': ap_70,
                        'mpre_50': mpre_50,
                        'mrec_50': mrec_50,
                        'mpre_70': mpre_70,
                        'mrec_70': mrec_70,
                        })
        if infer_info is None:
            yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
        else:
            yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))

        print('class_{}:\n'.format(tpe),'The Average Precision at IOU 0.3 is %.2f, '
            'The Average Precision at IOU 0.5 is %.2f, '
            'The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70), '\n')

    return ap_30, ap_50, ap_70

def eval_final_results(result_stat, save_path, num_agent, file_handle):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    print(f"0.3: tp({sum(result_stat[0.3]['tp'])}), fp({sum(result_stat[0.3]['fp'])})")
    print(f"0.5: tp({sum(result_stat[0.5]['tp'])}), fp({sum(result_stat[0.5]['fp'])})")
    print(f"0.7: tp({sum(result_stat[0.7]['tp'])}), fp({sum(result_stat[0.7]['fp'])})")

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    if num_agent == 0:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_average.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_agent{num_agent}.yaml'))

    print('------------------------------------------------------------------------------')
    file_handle.write('------------------------------------------------------------------------------ \n')
    if num_agent == 0:
        print('For all samples: ')
        file_handle.write('For all samples: \n')
    else:
        print(f"For samples whose agent num is {num_agent}: ")
        file_handle.write(f"For samples whose agent num is {num_agent}:  \n")
    print('The Average Precision at IOU 0.3 is %.4f, '
          'The Average Precision at IOU 0.5 is %.4f, '
          'The Average Precision at IOU 0.7 is %.4f' % (ap_30, ap_50, ap_70))
    file_handle.write('The Average Precision at IOU 0.3 is %.4f, '
          'The Average Precision at IOU 0.5 is %.4f, '
          'The Average Precision at IOU 0.7 is %.4f' % (ap_30, ap_50, ap_70) + '\n')
