from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .autoencoder import Autoencoder, Normalize
from .utils import label_attacker, rm_com_pair

RESIDUAL_AE = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_residual/model/autoencoder.pth"
DATA_CENTER = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_residual/data_center_residual.npy"
# CALIBRATION_FILE = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_residual/reconstruction_losses_residual.npy"
CALIBRATION_FILE = "/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/mreconstruction_loss_validation_v1.npy"


class ResidualAutoEncoderReconstructionLoss(nn.Module):
    def __init__(self, checkpoint_path=RESIDUAL_AE, data_center=DATA_CENTER):
        super().__init__()
        self.ae = Autoencoder(base_channel_size=32, latent_dim=256, num_input_channels=256)
        self.ae.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        self.normalize = Normalize(mean=np.load(data_center), std=1.0)

    @torch.no_grad()
    def forward(self, residual_feat:torch.Tensor):
        normed_feat = self.normalize(residual_feat)
        recon, _ = self.ae(normed_feat)
        recon_loss = F.mse_loss(normed_feat, recon, reduction="none").sum(dim=(1,2,3))
        return recon_loss

class ResidualAutoEncoderDetector(nn.Module):
    def __init__(self, checkpoint_path=RESIDUAL_AE, data_center=DATA_CENTER):
        super().__init__()
        # self.ae = Autoencoder(base_channel_size=32, latent_dim=256, num_input_channels=256)
        # self.ae.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        # self.normalize = Normalize(mean=np.load(data_center), std=1.0)
        self.loss = ResidualAutoEncoderReconstructionLoss(checkpoint_path, data_center)

        calibration_set = np.load(CALIBRATION_FILE)
        self.threshold = np.percentile(calibration_set, 95)
    
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
                batch_size = 1,):
        """
        input:
            model and model input
        output:
           score, com_src, com_tgt, total_tested, correct 
        """
        # encode feature
        x_0, x_1, x_2, x_3, x_4 = model.encode(bev)

        # select features to communicate and fuse
        feat_maps, size = model.select_com_layer(x_0, x_1, x_2, x_3, x_4)

        # com pairs for attack detection
        com_srcs, com_tgts = model.get_attack_det_com_pairs(num_agent)
        com_src_to_det = torch.cat([com_srcs[i][1::2] for i in range(1, len(com_srcs))], dim=0)
        com_tgt_to_det = torch.cat([com_tgts[i][1::2] for i in range(1, len(com_tgts))], dim=0)

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

        # normed_feat = self.normalize(residual_feat)
        # recon, _ = self.ae(normed_feat)
        # recon_loss = F.mse_loss(normed_feat, recon, reduction="none").sum(dim=(1,2,3))
        recon_loss = self.loss(residual_feat)

        is_attacker = recon_loss > self.threshold
        detected_src = com_src_to_det[is_attacker]
        detected_tgt = com_tgt_to_det[is_attacker]

        com_src, com_tgt = model.get_default_com_pair(num_agent)
        com_src, com_tgt = rm_com_pair(com_src, com_tgt, detected_src, detected_tgt)
        # print(attack_src, attack_tgt)
        # print(com_src, com_tgt)

        attacker_label = label_attacker(com_src_to_det, com_tgt_to_det, attack_src, attack_tgt)

        total = len(attacker_label)
        correct = (attacker_label == is_attacker).sum().item()

        return com_src, com_tgt, total, correct, \
            {"score": recon_loss.detach().cpu().numpy(), 
             "label": attacker_label.detach().cpu().numpy(), 
             "pred": is_attacker.long().detach().cpu().numpy()}