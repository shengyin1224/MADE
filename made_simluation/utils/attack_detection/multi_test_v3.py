from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from .bh_procedure import build_bh_procedure
from .residual_autoencoder import ResidualAutoEncoderReconstructionLoss
from .residual_autoencoder_v0 import ResidualAutoEncoderV0ReconstructionLoss
from .residual_autoencoder_v2 import ResidualAutoEncoderV2ReconstructionLoss
from .raw_autoencoder import RawAutoEncoderReconstructionLoss
from .utils import label_attacker, rm_com_pair

RAW_CALIBRATION_FILE = "../calibration/reconstruction_losses_raw.npy"
# RESIDUAL_CALIBRATION_FILE = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_residual/reconstruction_losses_residual.npy"
RESIDUAL_CALIBRATION_FILE = "../calibration/mreconstruction_loss_validation_v1.npy"

RESIDUAL_CALIBRATION_FILE_V2 = "mreconstruction_loss_validation_v2.npy"

LOAD_DICT_BASE = {
    "residual_ae": "experiments/residual_ae_v2/",
    "match_cost": "experiments/match_cost_v2/",
}

class MultiTestDetector(nn.Module):
    def __init__(self, match_cost=True, residual_ae=True, raw_ae=False, residual_ae_v0=False, load_score:Dict={}, multi_test_alpha=0.05, ):
        super().__init__()
        self.match_cost = match_cost
        self.residual_ae = residual_ae
        self.raw_ae = raw_ae

        dists = []
        if match_cost:
            dists.append(np.load("match_costs_validation.npy"))
            if load_score != None and "match_cost" in load_score and os.path.exists(os.path.join(load_score['match_cost'], "result.pkl")):
                with open(os.path.join(load_score['match_cost'], "result.pkl"), 'rb') as f:
                    result = pickle.load(f)
                score = result['score']
                self.load_match_cost = True
                self.saved_match_cost = score
                print("Load match cost score")
            else:
                self.load_match_cost = False
                
        if residual_ae:
            # V1
            # self.residual_recon_loss = ResidualAutoEncoderReconstructionLoss()
            # dists.append(np.load(RESIDUAL_CALIBRATION_FILE))
            # V2
            self.residual_recon_loss = ResidualAutoEncoderV2ReconstructionLoss()
            dists.append(np.load(RESIDUAL_CALIBRATION_FILE_V2))
            if load_score != None and "residual_ae" in load_score and os.path.exists(os.path.join(load_score['residual_ae'], "result.pkl")):
                with open(os.path.join(load_score['residual_ae'], "result.pkl"), 'rb') as f:
                    result = pickle.load(f)
                score = result['score']
                self.load_residual_ae = True
                self.saved_residual_ae = score
                print("Load residual ae score")
            else:
                self.load_residual_ae = False

        if raw_ae:
            self.raw_recon_loss = RawAutoEncoderReconstructionLoss()
            dists.append(np.load(RAW_CALIBRATION_FILE))
            if load_score != None and "residual_ae" in load_score and os.path.exists(os.path.join(load_score['residual_ae'], "result.pkl")):
                with open(os.path.join(load_score['residual_ae'], "result.pkl"), 'rb') as f:
                    result = pickle.load(f)
                score = result['score']
                self.load_raw_ae = True
                self.saved_raw_ae = score
            else:
                self.load_raw_ae = False
        
        self.bh_test = build_bh_procedure(dists=dists, fdr=multi_test_alpha)
        # import ipdb;ipdb.set_trace()

    @torch.no_grad()
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
        # encode feature
        x_0, x_1, x_2, x_3, x_4 = model.encode(bev)

        # select features to communicate and fuse
        feat_maps, size = model.select_com_layer(x_0, x_1, x_2, x_3, x_4)

        # com pairs for attack detection
        com_srcs, com_tgts = model.get_attack_det_com_pairs(num_agent)
        com_src_to_det = torch.cat([com_srcs[i][1::2] for i in range(1, len(com_srcs))], dim=0)
        com_tgt_to_det = torch.cat([com_tgts[i][1::2] for i in range(1, len(com_tgts))], dim=0)

        scores_for_test = []

        if self.match_cost:
            if self.load_match_cost:
                scores_for_test.append(self.saved_match_cost[self.cnt])
            else:
                results_list = model.multi_com_forward(
                    bev, trans_matrices, com_srcs, com_tgts, attack, attack_src, attack_tgt, batch_size)
                k = num_agent[0, 0]
                box_list = [model.post_process(results, anchors, k) for results in results_list]

                ego_result = box_list[0]
                match_cost = [model.matcher(box_list[i], ego_result) for i in range(1, len(box_list))]

                match_costs_tensor = torch.Tensor(match_cost).to(bev.device)
                match_costs_tensor = match_costs_tensor[:, 0] if match_costs_tensor.ndim == 3 else match_costs_tensor
                match_costs_tensor = match_costs_tensor.reshape(-1)
                scores_for_test.append(match_costs_tensor.detach().cpu().numpy())

        if self.residual_ae:
            if self.load_residual_ae:
                scores_for_test.append(self.saved_residual_ae[self.cnt])
            else:
                fused_features = []
                for com_src, com_tgt in zip(com_srcs, com_tgts):
                    # fuse feature
                    if attack is not None:
                        perturb = model.place_attack_v2(
                            attack, attack_src, attack_tgt, com_src, com_tgt)
                        fused_feat = model.communication_attack(
                                    feat_maps, trans_matrices, com_src, com_tgt, size, perturb, batch_size)
                    else:
                        fused_feat = model.communication_v2(
                            feat_maps, trans_matrices, com_src, com_tgt, size, batch_size
                        )
                    fused_features.append(fused_feat[:num_agent[0, 0]])
                
                residual_feat = [fused_features[i] - fused_features[0] for i in range(1, len(fused_features))]
                residual_feat = torch.cat(residual_feat, dim=0)
                residual_recon_loss = self.residual_recon_loss(residual_feat)
                scores_for_test.append(residual_recon_loss.detach().cpu().numpy())
        
        if self.raw_ae:
            if self.load_raw_ae:
                scores_for_test.append(self.saved_raw_ae[self.cnt])
            else:
                N, C, H, W = feat_maps.shape
                feat_maps = feat_maps.reshape(batch_size, -1, C, H, W)

                if attack is not None:
                    perturb = model.place_attack_v2(attack, attack_src, attack_tgt, com_src_to_det, com_tgt_to_det)

                    feat_to_det = feat_maps[com_src_to_det[:, 0], com_src_to_det[:, 1]] + perturb
                else:
                    feat_to_det = feat_maps[com_src_to_det[:, 0], com_src_to_det[:, 1]]
                raw_recon_loss = self.raw_recon_loss(feat_to_det)
                scores_for_test.append(raw_recon_loss.detach().cpu().numpy())

        scores_for_test = np.stack(scores_for_test, axis=1)

        is_attacker = []
        for i, score in enumerate(scores_for_test):
            rejected = self.bh_test.test_v2(score)
            is_attacker.append(rejected)
        is_attacker = torch.Tensor(is_attacker).bool().to(com_src_to_det.device)

        # import pdb; pdb.set_trace()
        detected_src = com_src_to_det[is_attacker]
        detected_tgt = com_tgt_to_det[is_attacker]

        com_src, com_tgt = model.get_default_com_pair(num_agent)
        com_src, com_tgt = rm_com_pair(com_src, com_tgt, detected_src, detected_tgt)

        attacker_label = label_attacker(com_src_to_det, com_tgt_to_det, attack_src, attack_tgt)

        total = len(attacker_label)
        correct = (attacker_label == is_attacker).sum().item()

        return com_src, com_tgt, total, correct, \
            {"score": scores_for_test, 
             "label": attacker_label.detach().cpu().numpy(), 
             "pred": is_attacker.long().detach().cpu().numpy()}