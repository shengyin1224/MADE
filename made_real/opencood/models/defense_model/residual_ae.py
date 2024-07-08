from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .autoencoder import UNet
from .base_att_det_model import BaseAttackerDetectModel, regroup

class ResidualAEDetector(BaseAttackerDetectModel):
    def __init__(self, channels=64, ckpt=None, cali_file=None, threshold = 0):
        super().__init__()
        
        self.channels = channels
        self.unet = UNet(channels, channels)
        # self.unet = UNet_Double(channels, channels)
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location='cpu')
            self.unet.load_state_dict(state_dict)
        self.unet.eval()
        
        if cali_file is not None:
            raise NotImplementedError()
        self.threshold = threshold
    
    def forward(self,
                feat_map_list: List[torch.Tensor], 
                record_len: torch.Tensor, 
                t_matrix: List[torch.Tensor],
                feat_src: List[torch.Tensor], 
                gt_attacker_label: List[torch.Tensor]=None):
        """
        args:
            feat_map_list
            record_len
            feat_src: 每个batch选哪几个id的feature来测
            gt_attacker_label: False->not attacker, True->attacker
        return:
            id_keep,
            additional_dict
        """
        additional_dict = {}
        
        feature_map = feat_map_list

        import ipdb; ipdb.set_trace()
        recon = self.unet(feature_map).squeeze[:,0]
        recon_loss = F.mse_loss(recon, feature_map, reduction="none").sum(dim=[1,2,3])

        is_attacker = recon_loss > self.threshold
        batch_is_attacker = regroup(is_attacker, 
                                    record_len=record_len.new_tensor([len(s) for s in feat_src]))
        id_keep = [torch.cat([s.new_zeros((1,)), s[~att]]) 
                   for s, att in zip(feat_src, batch_is_attacker)]
        id_rm = [s[att] for s, att in zip(feat_src, batch_is_attacker)]
        correct = (is_attacker == torch.cat(gt_attacker_label)).sum().item()

        additional_dict['score'] = recon_loss.detach().cpu().numpy()
        additional_dict['correct'] = correct
        additional_dict['total'] = len(is_attacker)

        return id_keep, additional_dict
