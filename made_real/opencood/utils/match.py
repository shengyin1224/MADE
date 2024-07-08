from typing import Dict, List
import torch 
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('/GPFS/data/shengyin/DAMC-HPC/Rotated_IoU')
from oriented_iou_loss import cal_giou_3d, cal_iou_3d

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class = 1, cost_giou = 1):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_boxes: Dict, target_boxes: Dict):
        """Match predicted boxes, only support single class now.

            pred_boxes: ego与某个agent结合算出的结果
            target_boxes: ego算出的结果/ego的GT

            boxes包含(box_tensor, score), 二者经过了nms处理
            box_tensor: (n,7)
            score: (n,)

            return:
                match cost
        """
        match_cost = []
        mean_score_cost = []
        mean_box_cost = []
        
        if pred_boxes['box_tensor'] == None:
            m = 0
        else:
            m = pred_boxes['box_tensor'].shape[0]
            # 候选框的数目进行限制
            if m > 50:
                m = 50
        if target_boxes['box_tensor'] == None:
            n = 0
        else:
            n = target_boxes['box_tensor'].shape[0]
        cost_mat = torch.zeros((max(m, n), max(m, n)))  
        score_cost = torch.zeros((max(m, n), max(m, n)))
        box_cost = torch.zeros((max(m, n), max(m, n)))
        
        if m > n:
            score_cost[:, n:] = 0
            box_cost[:, n:] = 0
        elif m < n:
            score_cost[m:, :] = target_boxes['score']
            box_cost[m:, :] = 1.0
        

        # 考虑每个元素之间的值
        for i in range(m):
            for j in range(n):
            # 由于score_cost[i].shape >= n
            # import ipdb; ipdb.set_trace()

                score_cost[i, j] = torch.maximum(torch.tensor(0), target_boxes['score'][j] - pred_boxes['score'][i])

                # import pdb; pdb.set_trace()
                ious = cal_giou_3d(pred_boxes['box_tensor'][i].unsqueeze(0).unsqueeze(0), target_boxes['box_tensor'][j].unsqueeze(0).unsqueeze(0))
                # import pdb; pdb.set_trace()
                box_cost[i, j] = ious[0].squeeze(0)

        # cost_mat = box_cost
        cost_mat = self.cost_class * score_cost + self.cost_giou * box_cost

        # 完成bbox一一匹配
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        match_cost.append((cost_mat[row_ind, col_ind].sum()) / n)
        mean_score_cost.append(score_cost[row_ind, col_ind].sum() / n)
        mean_box_cost.append((box_cost)[row_ind, col_ind].sum() / n)
        
        return match_cost, mean_score_cost, mean_box_cost
            

if __name__ == "__main__":
    matcher = HungarianMatcher()
    pred_tensor = np.load('/GPFS/data/shengyin/OpenCOOD-main/test_match/target_tensor.npy')
    pred_score = np.load('/GPFS/data/shengyin/OpenCOOD-main/test_match/target_score.npy')
    target_tensor = np.load('/GPFS/data/shengyin/OpenCOOD-main/test_match/target_tensor.npy')
    target_score = np.load('/GPFS/data/shengyin/OpenCOOD-main/test_match/target_score.npy')
    pred_tensor = torch.tensor(pred_tensor).cuda()
    # pred_score = torch.tensor(pred_score).cuda()
    target_tensor = torch.tensor(target_tensor).cuda()
    # target_score = torch.tensor(target_score).cuda()

    pred = {'box_tensor':pred_tensor,'score':pred_score}
    target = {'box_tensor':target_tensor,'score':target_score}
    match_cost, mean_score_cost, mean_box_cost = matcher(pred, target)
    print(match_cost, mean_score_cost, mean_box_cost)
