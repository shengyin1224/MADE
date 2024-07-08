import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from .utils import label_attacker, rm_com_pair
LOAD_SCORE_BASE = "/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/match_cost_v2/"
class MatchCostDetector(nn.Module):
    def __init__(self, match_para = 1, load_score: str = None):
        super().__init__()
        # self.percentiles = {
        #     50: 0.536744492804809,
        #     60: 0.5839201450420946,
        #     70: 0.6304444880969823,
        #     80: 0.6880368602735208,
        #     90: 0.7671696103588593,
        #     95: 0.8326803338450066,}
        # self.threshold = self.percentiles[95]
        calibration_set = np.load(f"/GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/match_cost/match_costs_validation_para_{float(match_para)}.npy")
        # calibration_set = np.load(f"/GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/match_cost/match_costs_validation_para_1.npy")
        self.threshold = np.percentile(calibration_set, 95)
        if load_score is not None and os.path.exists(os.path.join(load_score, "result.pkl")):
            with open(os.path.join(load_score, "result.pkl"), 'rb') as f:
                result = pickle.load(f)
            score = result['score']
            self.load_score = True
            self.saved_score = score
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
                batch_size = 1, cnt = 0):
        
        self.cnt = cnt
        com_srcs, com_tgts = model.get_attack_det_com_pairs(num_agent)
        results_list = model.multi_com_forward(
            bev, trans_matrices, com_srcs, com_tgts, attack, attack_src, attack_tgt, batch_size)
        
        com_srcs_to_det = torch.cat([com_srcs[i][1::2] for i in range(1, len(com_srcs))], dim=0)
        com_tgts_to_det = torch.cat([com_tgts[i][1::2] for i in range(1, len(com_tgts))], dim=0)
        attacker_label = label_attacker(com_srcs_to_det, com_tgts_to_det, attack_src, attack_tgt)

        self.load_score = False
        if self.load_score:
            score = self.saved_score[self.cnt]
            match_costs_tensor = torch.Tensor(score).to(bev.device)
        else:
            k = num_agent[0, 0]
            box_list = [model.post_process(results, anchors, k) for results in results_list]

            # import pdb; pdb.set_trace()
            ego_result = box_list[0]
            match_cost = [model.matcher(box_list[i], ego_result) for i in range(1, len(box_list))]

            match_costs_tensor = torch.Tensor(match_cost).to(bev.device)
            match_costs_tensor = match_costs_tensor[:, 0] if match_costs_tensor.ndim == 3 else match_costs_tensor
            match_costs_tensor = match_costs_tensor.reshape(-1)

        is_attacker = match_costs_tensor > self.threshold
        # import pdb; pdb.set_trace()
        detected_src = com_srcs_to_det[is_attacker]
        detected_tgt = com_tgts_to_det[is_attacker]

        total = len(attacker_label)
        correct = (attacker_label== is_attacker).sum().item()

        com_src, com_tgt = model.get_default_com_pair(num_agent)
        com_src, com_tgt = rm_com_pair(com_src, com_tgt, detected_src, detected_tgt)

        return com_src, com_tgt, total, correct, \
            {"score": match_costs_tensor.detach().cpu().numpy(), 
             "label": attacker_label.detach().cpu().numpy(), 
             "pred": is_attacker.long().detach().cpu().numpy(),
            #  "bboxes": box_list,
            #  "match_cost": match_cost,
             }