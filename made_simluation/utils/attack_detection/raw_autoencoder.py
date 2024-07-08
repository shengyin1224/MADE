import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .autoencoder import Autoencoder, Normalize
from .utils import label_attacker, rm_com_pair

RAW_AE = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_raw_features/model/autoencoder.pth"
DATA_CENTER = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_raw_features/data_center.npy"
# CALIBRATION_FILE = "/DB/data/yanghengzhao/adversarial/DAMC/zhenxiang/autoencoder_raw_features/reconstruction_losses_raw.npy"
CALIBRATION_FILE = "/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/mreconstruction_loss_validation_raw.npy"

class RawAutoEncoderReconstructionLoss(nn.Module):
    def __init__(self, checkpoint_path=RAW_AE, data_center=DATA_CENTER):
        super().__init__()
        self.ae = Autoencoder(base_channel_size=32, latent_dim=256, num_input_channels=256)
        self.ae.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.ae.eval()
        
        self.normalize = Normalize(mean=np.load(data_center), std=1.0)

    @torch.no_grad()
    def forward(self, feat:torch.Tensor):
        normed_feat = self.normalize(feat)
        recon, _ = self.ae(normed_feat)
        recon_loss = F.mse_loss(normed_feat, recon, reduction="none").sum(dim=(1,2,3))
        return recon_loss

class RawAutoEncoderDetector(nn.Module):
    def __init__(self, checkpoint_path=RAW_AE, data_center=DATA_CENTER):
        super().__init__()
        self.ae = Autoencoder(base_channel_size=32, latent_dim=256, num_input_channels=256)
        self.ae.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        self.normalize = Normalize(mean=np.load(data_center), std=1.0)

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
        # com_src, com_tgt = model.get_default_com_pair(num_agent)
        # com_src_to_det, com_tgt_to_det = rm_self_loop(com_src, com_tgt)
        com_srcs, com_tgts = model.get_attack_det_com_pairs(num_agent)
        com_src_to_det = torch.cat([com_srcs[i][1::2] for i in range(1, len(com_srcs))], dim=0)
        com_tgt_to_det = torch.cat([com_tgts[i][1::2] for i in range(1, len(com_tgts))], dim=0)

        N, C, H, W = feat_maps.shape
        feat_maps = feat_maps.reshape(batch_size, -1, C, H, W)

        if attack is not None:
            perturb = model.place_attack_v2(attack, attack_src, attack_tgt, com_src_to_det, com_tgt_to_det)

            feat_to_det = feat_maps[com_src_to_det[:, 0], com_src_to_det[:, 1]] + perturb
        else:
            feat_to_det = feat_maps[com_src_to_det[:, 0], com_src_to_det[:, 1]]
        
        normed_feat = self.normalize(feat_to_det)
        recon, _ = self.ae(normed_feat)
        recon_loss = F.mse_loss(normed_feat, recon, reduction="none").sum(dim=(1,2,3))

        is_attacker = recon_loss > self.threshold
        detected_src = com_src_to_det[is_attacker]
        detected_tgt = com_tgt_to_det[is_attacker]

        com_src, com_tgt = model.get_default_com_pair(num_agent)
        com_src, com_tgt = rm_com_pair(com_src, com_tgt, detected_src, detected_tgt)

        attacker_label = label_attacker(com_src_to_det, com_tgt_to_det, attack_src, attack_tgt)

        total = len(attacker_label)
        correct = (attacker_label == is_attacker).sum().item()

        return com_src, com_tgt, total, correct, \
            {"score": recon_loss.detach().cpu().numpy(), 
             "label": attacker_label.detach().cpu().numpy(), 
             "pred": is_attacker.long().detach().cpu().numpy()}



def rm_self_loop(com_src, com_tgt):
    new_com_src, new_com_tgt = [], []
    for src, tgt in zip(com_src, com_tgt):
        if not (src == tgt).all():  # not self loop
            new_com_src.append(src)
            new_com_tgt.append(tgt)
    return torch.stack(new_com_src, dim=0), torch.stack(new_com_tgt, dim=0)
