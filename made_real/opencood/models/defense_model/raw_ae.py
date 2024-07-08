from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .autoencoder import UNet
from .base_att_det_model import BaseAttackerDetectModel, regroup

class RawAEDetector(BaseAttackerDetectModel):
    def __init__(self, channels=64, ckpt=None, cali_file=None):
        super().__init__()
        self.channels = channels
        self.unet = UNet(channels, channels)
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location='cpu')
            self.unet.load_state_dict(state_dict)
        
        if cali_file is not None:
            raise NotImplementedError()
        else:
            self.threshold = 0.0
    
    def forward(self, 
                feat_map: torch.Tensor, 
                record_len: torch.Tensor, 
                feat_src: List[torch.Tensor], 
                gt_attacker_label: List[torch.Tensor]=None):
        """
        args:
            feat_map 可以从 feature_list中得到
            record_len
            feat_src: 每个batch选哪几个id的feature来测
            gt_attacker_label: False->not attacker, True->attacker
        return:
            id_keep,
            additional_dict
        """
        assert feat_map.shape[0] == sum(record_len)
        additional_dict = {}
        batch_feat_maps = regroup(feat_map, record_len)
        input_fm = torch.cat([fm[s] for fm, s in zip(batch_feat_maps, feat_src)], dim=0)
        recon = self.unet(input_fm)[:, 0]
        recon_loss = F.mse_loss(recon, input_fm, reduction="none").sum(dim=(1,2,3))

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


def unit_test():
    detector = RawAEDetector(64)
    feat_map = torch.Tensor(5, 64, 100, 252)
    id_keep, additional_dict = detector(feat_map, 
                                        torch.tensor([2,3]).long(), 
                                        [torch.arange(1,2), torch.arange(1, 3)], 
                                        [torch.zeros((1,)).bool(), torch.zeros((2,)).bool()])
    import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    unit_test()