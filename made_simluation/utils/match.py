from typing import List
import torch 
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

from .iou_utils import cal_giou
"""
    match score: 
"""

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class = 1, cost_giou = 1):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_boxes: List, target_boxes: List):
        """Match predicted boxes, only support single class now.

            len(pred_boxes) == len(target_boxes) == num_samples

            pred_boxes[0] = [{"pred": ...,
                            "score": ...,
                            "rot_box": ...}]
            target_boxes[0] = [{"target": ...,
                            "score": ...,
                            "rot_box": ...}]

            return:
                match cost
        """
        match_cost = []
        mean_score_cost = []
        mean_box_cost = []
        # import pdb; pdb.set_trace()
        for pred, target in zip(pred_boxes, target_boxes):
            m = pred[0]['rot_box'].shape[0]
            n = target[0]['rot_box'].shape[0]
            cost_mat = np.zeros((max(m, n), max(m, n)))  # (pred, target)
            score_cost = np.zeros((max(m, n), max(m, n)))
            box_cost = np.zeros((max(m, n), max(m, n)))
            
            if m > n:
                # cost_mat[:, n:] = 0
                score_cost[:, n:] = 0
                box_cost[:, n:] = 0
            elif m < n:
                # cost_mat[m:, :] = self.cost_class * target[0]['score'] + self.cost_giou * 1 
                score_cost[m:, :] = target[0]['score']
                box_cost[m:, :] = 1.0

            for i in range(m):
                for j in range(n):
                    # cost_mat[i, j] = self.cost_class * np.maximum(0, target[0]['score'][j] - pred[0]['score'][i]) + \
                        # self.cost_giou * cal_giou(target[0]['rot_box'][None, j:j+1], pred[0]['rot_box'][None, i:i+1])[0].cpu().numpy()
                    score_cost[i, j] = np.maximum(0, target[0]['score'][j] - pred[0]['score'][i])
                    box_cost[i, j] = cal_giou(target[0]['rot_box'][None, j:j+1], pred[0]['rot_box'][None, i:i+1])[0].cpu().numpy()
            
            cost_mat = self.cost_class * score_cost + self.cost_giou * box_cost
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            match_cost.append(cost_mat[row_ind, col_ind].sum() / n)
            mean_score_cost.append(score_cost[row_ind, col_ind].sum() / n)
            mean_box_cost.append(box_cost[row_ind, col_ind].sum() / n)
        
        return match_cost, mean_score_cost, mean_box_cost
            
class HungarianMatcherV2(nn.Module):
    def __init__(self, cost_class = 1, cost_giou = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_boxes: List, target_boxes: List):
        """Match predicted boxes, only support single class now.

            len(pred_boxes) == len(target_boxes) == num_samples

            pred_boxes[0] = [{"pred": ...,
                            "score": ...,
                            "rot_box": ...}]
            target_boxes[0] = [{"target": ...,
                            "score": ...,
                            "rot_box": ...}]

            return:
                match cost
        """
        match_cost = []
        mean_score_cost = []
        mean_box_cost = []
        for pred, target in zip(pred_boxes, target_boxes):
            m = pred[0]['rot_box'].shape[0]
            n = target[0]['rot_box'].shape[0]
            cost_mat = np.zeros((max(m, n), max(m, n)))  # (pred, target)
            score_cost = np.zeros((max(m, n), max(m, n)))
            box_cost = np.zeros((max(m, n), max(m, n)))
            
            if m > n:
                # cost_mat[:, n:] = 0
                score_cost[:, n:] = pred[0]['score'][:, np.newaxis]
                box_cost[:, n:] = 1.0
            elif m < n:
                # cost_mat[m:, :] = self.cost_class * target[0]['score'] + self.cost_giou * 1 
                score_cost[m:, :] = 0.0
                box_cost[m:, :] = 0.0

            for i in range(m):
                for j in range(n):
                    # cost_mat[i, j] = self.cost_class * np.maximum(0, target[0]['score'][j] - pred[0]['score'][i]) + \
                        # self.cost_giou * cal_giou(target[0]['rot_box'][None, j:j+1], pred[0]['rot_box'][None, i:i+1])[0].cpu().numpy()
                    score_cost[i, j] = np.maximum(0, target[0]['score'][j] - pred[0]['score'][i])
                    box_cost[i, j] = cal_giou(target[0]['rot_box'][None, j:j+1], pred[0]['rot_box'][None, i:i+1])[0].cpu().numpy()
            
            cost_mat = self.cost_class * score_cost + self.cost_giou * box_cost
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            match_cost.append(cost_mat[row_ind, col_ind].sum() / m)
            mean_score_cost.append(score_cost[row_ind, col_ind].sum() / m)
            mean_box_cost.append(box_cost[row_ind, col_ind].sum() / m)
        
        return match_cost, mean_score_cost, mean_box_cost
            
