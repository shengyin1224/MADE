import torch
import torch.nn as nn
from typing import List

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

class BaseAttackerDetectModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def del_attacker(id_keep: List[torch.Tensor], 
                     feat_maps: List[torch.Tensor],
                     record_len: torch.Tensor,
                     t_matrix: List[torch.Tensor]):
        B = len(id_keep)
        assert B == record_len.shape[0]
        assert B == len(t_matrix)

        new_feat_maps = []
        for feat_map in feat_maps:
            batch_feat_map = regroup(feat_map, record_len)
            new_feat_maps.append(
                torch.cat([fm[keep] for fm, keep in zip(batch_feat_map, id_keep)])
            )
        
        new_t_matrix = []
        for b, keep in enumerate(id_keep):
            new_t_matrix.append(t_matrix[b][keep][:, keep])

        new_record_len = torch.tensor([len(k) for k in id_keep], 
                                      dtype=record_len.dtype, device=record_len.device)
        # max_len = new_record_len.max()

        return new_feat_maps, new_record_len, new_t_matrix

