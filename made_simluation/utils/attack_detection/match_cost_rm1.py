import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from .utils import label_attacker, rm_com_pair
from ..match import HungarianMatcherV2
LOAD_SCORE_BASE = "/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/match_cost_rm1/"

class MatchCostDetectorRM1(nn.Module):
    def __init__(self, load_score: str = None):
        super().__init__()
        calibration_set = np.load("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/match_costs_rm1_validation.npy")
        self.threshold = np.percentile(calibration_set, 95)
        if load_score is not None and os.path.exists(os.path.join(load_score, "result.pkl")):
            with open(os.path.join(load_score, "result.pkl"), 'rb') as f:
                result = pickle.load(f)
            score = result['score']
            self.load_score = True
            self.saved_score = score
            self.cnt = 0
        else:
            self.load_score = False
        self.load_score = False

        self.matcher = HungarianMatcherV2()

    def forward(self, 
                model: torch.nn.Module,
                bev: torch.Tensor,
                trans_matrices: torch.Tensor,
                num_agent: torch.Tensor,
                anchors: torch.Tensor,
                attack: torch.Tensor,
                attack_src: torch.Tensor,
                attack_tgt: torch.Tensor,
                batch_size = 1,):
        com_srcs, com_tgts, com_srcs_to_det, com_tgts_to_det = model.get_attack_det_com_pairs_rm1(num_agent)
        results_list = model.multi_com_forward(
            bev, trans_matrices, com_srcs, com_tgts, attack, attack_src, attack_tgt, batch_size)
        
        attacker_label = label_attacker(com_srcs_to_det, com_tgts_to_det, attack_src, attack_tgt)

        if self.load_score:
            score = self.saved_score[self.cnt]
            match_costs_tensor = torch.Tensor(score).to(bev.device)
            self.cnt += 1 
        else:
            k = num_agent[0, 0]
            box_list = [model.post_process(results, anchors, k) for results in results_list]

            ego_result = box_list[0]
            match_cost = [self.matcher(box_list[i], ego_result) for i in range(1, len(box_list))]

            match_costs_tensor = torch.Tensor(match_cost).to(bev.device)
            match_costs_tensor = match_costs_tensor[:, 0] if match_costs_tensor.ndim == 3 else match_costs_tensor
            match_costs_tensor = match_costs_tensor.reshape(-1)
            # match_costs_tensor[match_costs_tensor.isnan()] = 2.0

        is_attacker = match_costs_tensor > self.threshold
        detected_src = com_srcs_to_det[is_attacker]
        detected_tgt = com_tgts_to_det[is_attacker]

        total = len(attacker_label)
        # correct = 0
        correct = (attacker_label== is_attacker).sum().item()

        com_src, com_tgt = model.get_default_com_pair(num_agent)
        com_src, com_tgt = rm_com_pair(com_src, com_tgt, detected_src, detected_tgt)

        return com_src, com_tgt, total, correct, \
            {
            "score": match_costs_tensor.detach().cpu().numpy(), 
             "label": attacker_label.detach().cpu().numpy(), 
             "pred": is_attacker.long().detach().cpu().numpy(),
            #  "bboxes": box_list,
            #  "match_cost": match_cost,
            }