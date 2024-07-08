from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .autoencoder import UNet, Autoencoder
from .autoencoder1 import Autoencoder as Ae
from .base_att_det_model import BaseAttackerDetectModel, regroup

class RawAEDetector(BaseAttackerDetectModel):
    def __init__(self, channels=64, ckpt=None, cali_file=None, threshold = 0, layer_num = 0):
        super().__init__()
        self.channels = channels
        # self.unet = UNet(channels, channels)
        if layer_num == 0:
            self.unet = Autoencoder(base_channel_size=256, latent_dim=2048, num_input_channels=channels)
        elif layer_num == 1:
            self.unet = Ae(base_channel_size=256, latent_dim=2048, num_input_channels=channels)
        else:
            self.unet = UNet(channels, channels)
            
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location='cpu')
            self.unet.load_state_dict(state_dict)
        
        self.unet.eval()
        
        if cali_file is not None:
            raise NotImplementedError()
        self.threshold = threshold
    
    def forward(self, 
                feat_map: torch.Tensor, 
                record_len: torch.Tensor, 
                feat_src: List[torch.Tensor], 
                gt_attacker_label: List[torch.Tensor]=None,
                layer_num: int=0):
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
        # import ipdb; ipdb.set_trace()
        additional_dict = {}
        batch_feat_maps = regroup(feat_map, record_len) # batch_feat_maps[0] == feat_map
        input_fm = torch.cat([fm[s] for fm, s in zip(batch_feat_maps, feat_src)], dim=0) # input_fm[0] = feat_map[1]
        # input_fm = feat_map
        # recon = self.unet(input_fm)[:, 0]
        if layer_num == 0 or layer_num == 1:
            recon, feat = self.unet(input_fm)
        else:
            recon = self.unet(input_fm)[:, 0]
        if layer_num == 0 or layer_num == 2:
            recon_loss = F.mse_loss(recon, input_fm, reduction="none").sum(dim=(1,2,3))
        else:
            recon_loss = F.mse_loss(recon, input_fm[:, :, :49, :121], reduction="none").sum(dim=(1,2,3))

        is_attacker = recon_loss > self.threshold
        batch_is_attacker = regroup(is_attacker, 
                record_len=record_len.new_tensor([len(s) for s in feat_src])) # batch_is_attacker[0] = is_attacker[1]
        # id_keep shows which agent should be kept, id_rm shows which agent should be removed
        id_keep = [torch.cat([s.new_zeros((1,)), s[~att]]) 
                   for s, att in zip(feat_src, batch_is_attacker)]
        id_rm = [s[att] for s, att in zip(feat_src, batch_is_attacker)]
        # if detector is correct for this sample
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