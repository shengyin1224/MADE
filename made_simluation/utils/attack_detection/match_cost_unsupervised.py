import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import median_abs_deviation as MAD

from .utils import label_attacker, rm_com_pair
LOAD_SCORE_BASE = "/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/match_cost_unsupervised/"

class MatchCostUnsupervisedDetector(nn.Module):
    def __init__(self, load_score: str = None):
        super().__init__()
        self.threshold = 2
        if load_score is not None and os.path.exists(os.path.join(load_score, "result.pkl")):
            with open(os.path.join(load_score, "result.pkl"), 'rb') as f:
                result = pickle.load(f)
            # import ipdb;ipdb.set_trace()
            self.saved_score = result['score']
            # self.saved_match_cost_list = result['match_costs']
            # raise NotImplementedError()
            self.load_score = True
            self.cnt = 0
        else:
            self.load_score = False

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
        """
            Only support num_agent=4
        """            
        com_srcs, com_tgts = model.get_attack_det_com_pairs(num_agent)

        com_srcs_to_det = torch.cat([com_srcs[i][1::2] for i in range(1, len(com_srcs))], dim=0)
        com_tgts_to_det = torch.cat([com_tgts[i][1::2] for i in range(1, len(com_tgts))], dim=0)
        attacker_label = label_attacker(com_srcs_to_det, com_tgts_to_det, attack_src, attack_tgt)

        if self.load_score:
            all_score = self.saved_score[self.cnt]
            # match_cost_list = self.saved_match_cost_list[self.cnt]
            # # match_costs_tensor = torch.Tensor(score).to(bev.device)
            # # raise NotImplementedError()
            # score_list = []
            # for i, match_cost_np in enumerate(match_cost_list):
            #     med = np.median(match_cost_np.reshape(-1))
            #     mad = MAD(match_cost_np.reshape(-1))
            #     score = np.abs(match_cost_np[:, i] - med) / (mad * 1.4826)
            #     score_list.append(score)
            # all_score = np.concatenate(score_list)
            self.cnt += 1 
        else:
            k = num_agent[0, 0]
            score_list = []
            match_cost_list = []
            for i in range(k):
                if len(attack > 0):
                    selected_attack = attack[attack_tgt[:, 1] == i]
                    selected_attack_src = attack_src[attack_tgt[:, 1] == i]
                else:
                    selected_attack = attack
                    selected_attack_src = attack_src
                results_list = model.single_view_multi_com_forward(
                    bev, trans_matrices, com_srcs, com_tgts, selected_attack, selected_attack_src, batch_size
                )
                box_list = [model.post_process(results, anchors, k) for results in results_list]
                ego_result = box_list[0]
                match_cost = [model.matcher(box_list[i], ego_result) for i in range(1, len(box_list))]
                match_cost_np = np.array(match_cost)
                match_cost_np = match_cost_np[:, 0] if match_cost_np.ndim == 3 else match_cost_np

                med = np.median(match_cost_np.reshape(-1))
                mad = MAD(match_cost_np.reshape(-1))
                score = np.abs(match_cost_np[:, i] - med) / (mad * 1.4826)
                score_list.append(score)
                match_cost_list.append(match_cost_np)
            
            all_score = np.concatenate(score_list)
            # results_list = model.multi_com_forward(
            #     bev, trans_matrices, com_srcs, com_tgts, attack, attack_src, attack_tgt, batch_size)
            # box_list = [model.post_process(results, anchors, k) for results in results_list]

            # ego_result = box_list[0]
            # match_cost = [model.matcher(box_list[i], ego_result) for i in range(1, len(box_list))]

            # match_costs_tensor = torch.Tensor(match_cost).to(bev.device)
            # match_costs_tensor = match_costs_tensor[:, 0] if match_costs_tensor.ndim == 3 else match_costs_tensor
            # match_costs_tensor = match_costs_tensor.reshape(-1)

        # is_attacker = match_costs_tensor > self.threshold
        is_attacker = torch.Tensor(all_score).to(attacker_label.device) > self.threshold
        detected_src = com_srcs_to_det[is_attacker]
        detected_tgt = com_tgts_to_det[is_attacker]

        total = len(attacker_label)
        correct = (attacker_label == is_attacker).sum().item()

        com_src, com_tgt = model.get_default_com_pair(num_agent)
        com_src, com_tgt = rm_com_pair(com_src, com_tgt, detected_src, detected_tgt)

        return com_src, com_tgt, total, correct, \
            {"score": all_score, 
             "label": attacker_label.detach().cpu().numpy(), 
             "pred": is_attacker.long().detach().cpu().numpy(),
             "match_costs": match_cost_list
            #  "bboxes": box_list,
            #  "match_cost": match_cost,
             }