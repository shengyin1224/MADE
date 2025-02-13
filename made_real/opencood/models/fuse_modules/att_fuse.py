"""
Implementation of Attn Fusion
"""

import torch
import torch.nn as nn
import os

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine_simple, get_rotated_roi
from matplotlib import pyplot as plt
from icecream import ic
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

P_value = 0
C_list = [64, 128, 256]
H_list = [100, 50, 25]
W_list = [252, 126, 63]


def new_plot_sub_tensors(tensor, num, cls):
    # Convert the torch tensor to a NumPy array
    tensor_np = tensor.detach().cpu().numpy()

    # Create a 2x1 grid for subplots
    fig, axs = plt.subplots(2, 1, figsize=(18, 10), tight_layout=True)

    # mask
    mask = ((cls.sigmoid()[0][0] > 0.1) * 1.0).unsqueeze(0)
    if tensor_np.shape[0] *  tensor_np.shape[1] == 50 * 126:
        # 定义下采样
        downsample = nn.MaxPool2d(2, stride=2)
        mask = downsample(mask)
    elif tensor_np.shape[0] *  tensor_np.shape[1] == 25 * 63:
        # 定义下采样
        downsample = nn.MaxPool2d(2, stride=2)
        mask = downsample(downsample(mask))

    mask = mask.squeeze(0).detach().cpu().numpy()

    # Plot each sub-tensor in a separate subplot
    for i in range(2):
        for j in range(1):
            heatmap = axs[i].imshow(tensor_np[:,:,j,i] * mask, cmap='viridis')
            axs[i].set_title(f"Sub-tensor {i} to {j}")
            axs[i].axis('off')

    # Add a title for the whole figure
    position = fig.add_axes([0.92, 0.12, 0.015, .78 ]) #位置[左,下,右,上]
    cbar = fig.colorbar(heatmap, ax=[axs[0], axs[1]], cax=position)
    fig.suptitle(f"Sub-Tensors of Sample{num}", fontsize=16)

    # Display the plot
    plus_value = P_value
    save_path = 'outcome/attention_score/1016_Tmp20' + f'_ESP_Step80_shape_{tensor_np.shape[0]}x{tensor_np.shape[1]}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + f'/sample_{num}.png')
    plt.close()


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim, temperature = 20):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.temperature = temperature

    def forward(self, query, key, value, num, if_draw, cls, if_att_score):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)

        # add temperature variables
        # import ipdb; ipdb.set_trace()
        temperature = self.temperature
        # 目前仅在第一层添加Temperature
        if temperature != 1:
            score = score / temperature

        if if_draw == 'no_fuse':
            Lelouch = 0
        elif if_draw != 'no_fuse':
            
            # import ipdb; ipdb.set_trace()
            attn = F.softmax(score, -1)

        # if if_draw == True:
        #     if cls != None:
        #         if score.shape[0] == 100 * 252:
        #             new_plot_sub_tensors(attn.reshape(100, 252, 2, 2), num, cls)
        #         elif score.shape[0] == 50 * 126:
        #             new_plot_sub_tensors(attn.reshape(50, 126, 2, 2), num, cls)
        #         elif score.shape[0] == 25 * 63:
        #             new_plot_sub_tensors(attn.reshape(25, 63, 2, 2), num, cls)

                
        context = torch.bmm(attn, value)
        if if_att_score:
            return context, attn
        else:
            return context, None
        
class AttFusion(nn.Module):
    def __init__(self, args):
        super(AttFusion, self).__init__()

        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']
        self.att = ScaledDotProductAttention(args['in_channels'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, xx, record_len, pairwise_t_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = self.regroup(xx, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        batch_node_features = split_x
        # iteratively update the features for num_iteration times

        out = []
        # iterate each batch
        for b in range(B):

            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            updated_node_features = []

            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))

            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            out.append(h)

        out = torch.stack(out)
        
        return out


    # def forward_debug(self, x, origin_x, record_len, pairwise_t_matrix):
    #     """
    #     Fusion forwarding
    #     Used for debug and visualization

        
    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         input data, (sum(n_cav), C, H, W)

    #     origin_x: torch.Tensor
    #         pillars (sum(n_cav), C, H * downsample_rate, W * downsample_rate)
            
    #     record_len : list
    #         shape: (B)
            
    #     pairwise_t_matrix : torch.Tensor
    #         The transformation matrix from each cav to ego, 
    #         shape: (B, L, L, 4, 4) 
            
    #     Returns
    #     -------
    #     Fused feature.
    #     """
    #     from matplotlib import pyplot as plt

    #     _, C, H, W = x.shape
    #     B, L = pairwise_t_matrix.shape[:2]

    #     split_x = self.regroup(x, record_len)
    #     split_origin_x = self.regroup(origin_x, record_len)

    #     # (B,L,L,2,3)
    #     pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    #     pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
    #     pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
    #     pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
    #     pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2


    #     # (B*L,L,1,H,W)
    #     roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
    #     for b in range(B):
    #         N = record_len[b]
    #         for i in range(N):
    #             one_tensor = torch.ones((L,1,H,W)).to(x)
    #             roi_mask[b,i] = warp_affine_simple(one_tensor, pairwise_t_matrix[b][i, :, :, :],(H, W))

    #     batch_node_features = split_x
    #     # iteratively update the features for num_iteration times

    #     # visualize warped feature map
    #     for b in range(B):
    #         # number of valid agent
    #         N = record_len[b]
    #         # (N,N,4,4)
    #         # t_matrix[i, j]-> from i to j
    #         t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

    #         # update each node i
    #         i = 0 # ego
    #         mask = roi_mask[b, i, :N, ...]
    #         # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
    #         # Notice we put i one the first dim of t_matrix. Different from original.
    #         # t_matrix[i,j] = Tji
    #         neighbor_feature = warp_affine_simple(batch_node_features[b],
    #                                         t_matrix[i, :, :, :],
    #                                         (H, W))
    #         for idx in range(N):
    #             plt.imshow(torch.max(neighbor_feature[idx],0)[0].detach().cpu().numpy())
    #             plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/feature_{b}_{idx}")
    #             plt.clf()
    #             plt.imshow(mask[idx][0].detach().cpu().numpy())
    #             plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/mask_feature_{b}_{idx}")
    #             plt.clf()


        
    #     # visualize origin pillar feature 
    #     origin_node_features = split_origin_x

    #     for b in range(B):
    #         N = record_len[b]
    #         # (N,N,4,4)
    #         # t_matrix[i, j]-> from i to j
    #         t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

    #         i = 0 # ego
    #         # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
    #         # Notice we put i one the first dim of t_matrix. Different from original.
    #         # t_matrix[i,j] = Tji
    #         neighbor_feature = warp_affine_simple(origin_node_features[b],
    #                                         t_matrix[i, :, :, :],
    #                                         (H*self.downsample_rate, W*self.downsample_rate))

    #         for idx in range(N):
    #             plt.imshow(torch.max(neighbor_feature[idx],0)[0].detach().cpu().numpy())
    #             plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/origin_{b}_{idx}")
    #             plt.clf()