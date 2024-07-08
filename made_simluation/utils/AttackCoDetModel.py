from typing import List, Optional, Union
from detectron2.layers import batched_nms_rotated
from utils.detection_util import bev_box_decode_torch, center_to_corner_box2d_torch, sincos2deg, rescale_boxes
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model import *
import convolutional_rnn as convrnn
import torch_scatter as ts
from torch_geometric.utils import softmax
import numpy as np
from data.obj_util import center_to_corner_box2d
from detectron2.layers import nms_rotated
from utils.match import HungarianMatcher
from utils.robosac import HungarianMatcher_ROBOSAC
import os

class DiscoNet(nn.Module):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, match_para = 1):
        super(DiscoNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.kd_flag = kd_flag
        self.layer = layer
        self.ModulationLayer3 = ModulationLayer3(config)
        if self.layer == 3:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(256)
            self.com_size = (1, 256, 32, 32)
        elif self.layer == 2:
            self.PixelWeightedFusion = PixelWeightedFusionSoftmax(128)
            self.com_size = (1, 128, 64, 64)
        else:
            raise NotImplementedError(f"layer: {self.layer}")

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

        self.matcher = HungarianMatcher(cost_giou=match_para)
        self.get_jaccard_index = HungarianMatcher_ROBOSAC()
        print("DiscoNet from AttackCoDetModel")

    def agents2batch(self, feats):
        # TODO replace this with reshape
        # from (b, n, c, h, w) to (n*b, c, h, w)
        feat_mat = feats.transpose(0, 1).reshape(-1, *feats.shape[2:])
        return feat_mat

    def forward(self,
                bevs: torch.Tensor,
                trans_matrices: torch.Tensor,
                num_agent_tensor: torch.Tensor,
                batch_size: int = 1,
                com_src: Optional[torch.Tensor] = None,
                com_tgt: Optional[torch.Tensor] = None,
                attack: Optional[torch.Tensor] = None,
                attack_src: Optional[torch.Tensor] = None,
                attack_tgt: Optional[torch.Tensor] = None,
                batch_anchors: Optional[torch.Tensor] = None,
                nms: bool = False,
                *args, **kwargs):
        if "scene_name" in kwargs:
            self.scene_name = kwargs['scene_name']
        else:
            if hasattr(self, "scene_name"):
                del self.scene_name

        if com_src is None or com_tgt is None:
            assert num_agent_tensor is not None, "`num_agent_tensor`, `com_src` and `com_tgt` should not be None at the same time"
            com_src, com_tgt = self.get_default_com_pair(num_agent_tensor)

        results = self.com_forward_v2(bevs=bevs, trans_matrices=trans_matrices,
                                      com_src=com_src, com_tgt=com_tgt,
                                      attack=attack, attack_src=attack_src,
                                      attack_tgt=attack_tgt, batch_size=batch_size)

        if nms:
            assert batch_anchors is not None
            # non-empty agent num
            k = sum([torch.nonzero(b).shape[0] != 0 for b in bevs])
            return results, self.post_process(results, batch_anchors, k)
        else:
            return results

    def communication(self, feat_maps, trans_matrices, num_agent_tensor, size, batch_size=1):
        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        B, C, H, W = feat_maps.shape
        device = feat_maps.device
        local_com_mat = feat_maps.reshape(
            self.agent_num, batch_size, C, H, W).transpose(1, 0)  # (b, n ,c, h, w)
        local_com_mat_update = local_com_mat.clone()

        p = np.array([1.0, 0.0])

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]

                neighbor_feat_list = list()
                neighbor_feat_list.append(tg_agent)

                p_com_outage = np.random.choice([0, 1], p=p.ravel())

                if p_com_outage == 1:
                    agent_wise_weight_feat = neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            nb_agent = torch.unsqueeze(
                                local_com_mat[b, j], 0)  # [1 512 16 16]
                            nb_warp = all_warp[j]  # [4 4]
                            # normalize the translation vector
                            x_trans = (4*nb_warp[0, 3])/128
                            y_trans = -(4*nb_warp[1, 3])/128

                            theta_rot = torch.tensor([[nb_warp[0, 0], nb_warp[0, 1], 0.0], [
                                                     nb_warp[1, 0], nb_warp[1, 1], 0.0]]).type(dtype=torch.float).to(device)
                            theta_rot = torch.unsqueeze(theta_rot, 0)
                            grid_rot = F.affine_grid(
                                theta_rot, size=torch.Size(size))  # for grid sample

                            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(
                                dtype=torch.float).to(device)
                            theta_trans = torch.unsqueeze(theta_trans, 0)
                            grid_trans = F.affine_grid(
                                theta_trans, size=torch.Size(size))  # for grid sample

                            # first rotate the feature map, then translate it
                            warp_feat_rot = F.grid_sample(
                                nb_agent, grid_rot, mode='bilinear')
                            warp_feat_trans = F.grid_sample(
                                warp_feat_rot, grid_trans, mode='bilinear')
                            warp_feat = torch.squeeze(warp_feat_trans)
                            neighbor_feat_list.append(warp_feat)

                    neighbor_feats = torch.stack(neighbor_feat_list, dim=0)
                    cated_feats = torch.cat([tg_agent.unsqueeze(0).repeat(
                        neighbor_feats.shape[0], 1, 1, 1), neighbor_feats], dim=1)
                    agent_score = self.PixelWeightedFusion(cated_feats)
                    agent_weight = torch.softmax(agent_score, dim=0)
                    agent_wise_weight_feat = (
                        agent_weight * neighbor_feats).sum(dim=0)

                # feature update
                local_com_mat_update[b, i] = agent_wise_weight_feat

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        return feat_fuse_mat

    def communication_v2(self, feat_maps, trans_matrices, com_src, com_tgt, size, batch_size=1):
        """
            Support single agent or all agent communication
        """
        # com_src, com_tgt: [batch, agent_id] (N_pairs, 2)
        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        # num_agnet_tensor (b, n)

        B, C, H, W = feat_maps.shape
        local_com_mat = feat_maps.reshape(
            self.agent_num, batch_size, C, H, W).transpose(1, 0)  # (b, n ,c, h, w)
        local_com_mat_update = local_com_mat.clone()

        feat_src = local_com_mat[com_src[:, 0],
                                 com_src[:, 1]]  # (N_pairs, c, h, w)
        feat_tgt = local_com_mat[com_tgt[:, 0], com_tgt[:, 1]]

        # transformation to affine grid
        # (N_pairs, 4, 4)
        pair_trans_mat = trans_matrices[com_tgt[:,0], 
                                        com_tgt[:, 1], 
                                        com_src[:, 1]]
        theta = pair_trans_mat[:, [0, 0, 0, 1, 1, 1],
                               [0, 1, 3, 0, 1, 3]].reshape(-1, 2, 3)
        theta[:, 0, 2] = (4 * theta[:, 0, 2]) / 128
        theta[:, 1, 2] = -(4 * theta[:, 1, 2]) / 128
        theta = theta.float()

        theta_rot = theta.clone()
        theta_rot[:, :, 2] = 0
        grid_rot = F.affine_grid(
            theta_rot, size=torch.Size((theta.shape[0],) + size[1:]))
        warp_feat_rot = F.grid_sample(feat_src, grid_rot, mode='bilinear')

        theta_trans = theta.clone()
        theta_trans[:, [0, 1], [0, 1]] = 1
        theta_trans[:, [0, 1], [1, 0]] = 0
        grid_trans = F.affine_grid(
            theta_trans, size=torch.Size((theta.shape[0],) + size[1:]))
        warp_feat = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')

        cated_feats = torch.cat([feat_tgt, warp_feat], dim=1)
        agent_score = self.PixelWeightedFusion(
            cated_feats)  # (N_pairs, 1, h, w)

        reduce_tgt, cnt = torch.unique(com_tgt, return_counts=True, dim=0)
        indptr = torch.cumsum(
            torch.cat([cnt.new_zeros(size=(1,)), cnt]), dim=0)

        assert (ts.gather_csr(reduce_tgt, indptr) ==
                com_tgt).all(), "`com_tgt` should be ordered"
        normed_agent_score = softmax(agent_score, ptr=indptr, dim=0)
        reduced_feat = ts.segment_csr(
            warp_feat * normed_agent_score, indptr, reduce='sum')
        local_com_mat_update[reduce_tgt[:, 0], reduce_tgt[:, 1]] = reduced_feat

        # , warp_feat, agent_score, normed_agent_score, reduced_feat
        # import ipdb;ipdb.set_trace()
        if hasattr(self, "scene_name"):
            # np.save(os.path.join("vis_vis/attack/raw", self.scene_name), normed_agent_score.detach().cpu().numpy())
            save_dict = {
                "feature": feat_src.detach().cpu().numpy(),
                "fused": reduced_feat.detach().cpu().numpy(),
                "com_src": com_src.detach().cpu().numpy(),
                "com_tgt": com_tgt.detach().cpu().numpy(),
                "reduce_tgt": reduce_tgt.detach().cpu().numpy(),
            }
            import pickle
            with open(os.path.join("vis_vis2/no_attack/raw_raw", self.scene_name+".pkl"), 'wb') as f:
                pickle.dump(save_dict, f)
        return self.agents2batch(local_com_mat_update)

    def communication_attack(self, feat_maps, trans_matrices, com_src, com_tgt, size, perturb, batch_size=1):
        """
            Support single agent or all agent communication
            args:
                feat_maps: torch.Tensor [N, C, H, W]
                trans_matrices: torch.Tensor [B, 5, 5, 4, 4]
                com_src: torch.Tensor [n_pair, 2]
                com_tgt: torch.Tensor [n_pair, 2]
                size: torch.Size
                perturb: torch.Tensor [n_pair, C, H, W]
                batch_size: int 
        """

        B, C, H, W = feat_maps.shape
        local_com_mat = feat_maps.reshape(
            self.agent_num, batch_size, C, H, W).transpose(1, 0)  # (b, n ,c, h, w)
        local_com_mat_update = local_com_mat.clone()

        feat_src = local_com_mat[com_src[:, 0],
                                 com_src[:, 1]]  # (N_pairs, c, h, w)
        feat_tgt = local_com_mat[com_tgt[:, 0], com_tgt[:, 1]]

        # attack
        # import ipdb;ipdb.set_trace()
        feat_src = feat_src + perturb

        # transformation to affine grid
        # (N_pairs, 4, 4)
        pair_trans_mat = trans_matrices[com_tgt[:, 0],
                                        com_tgt[:, 1],
                                        com_src[:, 1]]
        # import ipdb;ipdb.set_trace()
        theta = pair_trans_mat[:, [0, 0, 0, 1, 1, 1],
                               [0, 1, 3, 0, 1, 3]].reshape(-1, 2, 3)
        theta[:, 0, 2] = (4 * theta[:, 0, 2]) / 128
        theta[:, 1, 2] = -(4 * theta[:, 1, 2]) / 128
        theta = theta.float()

        theta_rot = theta.clone()
        theta_rot[:, :, 2] = 0
        grid_rot = F.affine_grid(
            theta_rot, size=torch.Size((theta.shape[0],) + size[1:]))
        warp_feat_rot = F.grid_sample(feat_src, grid_rot, mode='bilinear')

        theta_trans = theta.clone()
        theta_trans[:, [0, 1], [0, 1]] = 1
        theta_trans[:, [0, 1], [1, 0]] = 0
        grid_trans = F.affine_grid(
            theta_trans, size=torch.Size((theta.shape[0],) + size[1:]))
        warp_feat = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')

        cated_feats = torch.cat([feat_tgt, warp_feat], dim=1)
        agent_score = self.PixelWeightedFusion(
            cated_feats)  # (N_pairs, 1, h, w)

        reduce_tgt, cnt = torch.unique(com_tgt, return_counts=True, dim=0)
        indptr = torch.cumsum(
            torch.cat([cnt.new_zeros(size=(1,)), cnt]), dim=0)

        assert (ts.gather_csr(reduce_tgt, indptr) ==
                com_tgt).all(), "`com_tgt` should be ordered"
        normed_agent_score = softmax(agent_score, ptr=indptr, dim=0)
        reduced_feat = ts.segment_csr(
            warp_feat * normed_agent_score, indptr, reduce='sum')

        local_com_mat_update[reduce_tgt[:, 0], reduce_tgt[:, 1]] = reduced_feat

        if hasattr(self, "scene_name"):
            # np.save(os.path.join("vis_vis/attack/raw", self.scene_name), normed_agent_score.detach().cpu().numpy())
            save_dict = {
                "feature": feat_src.detach().cpu().numpy(),
                "fused": reduced_feat.detach().cpu().numpy(),
                "com_src": com_src.detach().cpu().numpy(),
                "com_tgt": com_tgt.detach().cpu().numpy(),
                "reduce_tgt": reduce_tgt.detach().cpu().numpy(),
            }
            import pickle
            with open(os.path.join("vis_vis2/attack/raw_0319", self.scene_name+".pkl"), 'wb') as f:
                pickle.dump(save_dict, f)
        return self.agents2batch(local_com_mat_update)

    def single_communication(self, feat_maps, trans_matrices, num_agent_tensor, size, batch_size=1, agent_index=0, com=True):
        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        B, C, H, W = feat_maps.shape
        device = feat_maps.device
        local_com_mat = feat_maps.reshape(
            self.agent_num, batch_size, C, H, W).transpose(1, 0)
        # local_com_mat_update = local_com_mat.clone()
        # (b, 1, c, h, w)
        local_com_mat_update = local_com_mat[:,
                                             agent_index:agent_index+1].clone()

        if not com:
            # no communication
            return local_com_mat_update

        save_agent_weight_list = list()
        p = np.array([1.0, 0.0])

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, agent_index]
            if agent_index >= num_agent:
                continue
            i = agent_index
            tg_agent = local_com_mat[b, i]
            all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]

            neighbor_feat_list = list()
            neighbor_feat_list.append(tg_agent)

            #com_outage = random.randint(0,1)
            p_com_outage = np.random.choice([0, 1], p=p.ravel())

            if p_com_outage == 1:
                agent_wise_weight_feat = neighbor_feat_list[0]
            else:
                for j in range(num_agent):
                    # TODO remove this for iteration
                    if j != i:
                        nb_agent = torch.unsqueeze(
                            local_com_mat[b, j], 0)  # [1 512 16 16]
                        nb_warp = all_warp[j]  # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        # fail to merge two grid sample
                        theta_rot = torch.tensor([[nb_warp[0, 0], nb_warp[0, 1], 0.0], [
                                                 nb_warp[1, 0], nb_warp[1, 1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(
                            theta_rot, size=torch.Size(size))  # for grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(
                            dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(
                            theta_trans, size=torch.Size(size))  # for grid sample

                        # first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(
                            nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(
                            warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)
                        neighbor_feat_list.append(warp_feat)

                neighbor_feats = torch.stack(neighbor_feat_list, dim=0)
                cated_feats = torch.cat([tg_agent.unsqueeze(0).repeat(
                    neighbor_feats.shape[0], 1, 1, 1), neighbor_feats], dim=1)
                agent_score = self.PixelWeightedFusion(cated_feats)
                agent_weight = torch.softmax(agent_score, dim=0)
                agent_wise_weight_feat = (
                    agent_weight * neighbor_feats).sum(dim=0)

            # feature update
            local_com_mat_update[b, 0] = agent_wise_weight_feat

        return local_com_mat_update

    def select_com_layer(self,
                         x_0: torch.Tensor,
                         x_1: torch.Tensor,
                         x_2: torch.Tensor,
                         x_3: torch.Tensor,
                         x_4: torch.Tensor):
        if self.layer == 4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer == 3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)
        else:
            raise NotImplementedError(self.layer)
        return feat_maps, size

    def place_merged_feature_back(self, x_0: torch.Tensor,
                                  x_1: torch.Tensor,
                                  x_2: torch.Tensor,
                                  x_3: torch.Tensor,
                                  x_4: torch.Tensor,
                                  fused_feats: torch.Tensor):
        if self.layer == 4:
            return x_0, x_1, x_2, x_3, fused_feats
        elif self.layer == 3:
            return x_0, x_1, x_2, fused_feats, x_4
        elif self.layer == 2:
            return x_0, x_1, fused_feats, x_3, x_4
        elif self.layer == 1:
            return x_0, fused_feats, x_2, x_3, x_4
        elif self.layer == 0:
            return fused_feats, x_1, x_2, x_3, x_4
        else:
            raise NotImplementedError(self.layer)

    def encode(self, inputs):
        bevs = inputs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x_0, x_1, x_2, x_3, x_4 = self.u_encoder(bevs)
        return x_0, x_1, x_2, x_3, x_4

    def decode(self, x_0, x_1, x_2, x_3, x_4, batch_size):
        if self.kd_flag == 1:
            x_8, x_7, x_6, x_5 = self.decoder(
                x_0, x_1, x_2, x_3, x_4, batch_size, kd_flag=self.kd_flag)
            x = x_8
            return x, x_8, x_7, x_6, x_5
        else:
            x = self.decoder(x_0, x_1, x_2, x_3, x_4,
                             batch_size, kd_flag=self.kd_flag)
            return x

    def head_out(self, x):
        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0], -1, self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1, loc_preds.size(1), loc_preds.size(
            2), self.anchor_num_per_loc, self.out_seq_len, self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        # MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(
                0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(
                cls_preds.shape[0], -1, motion_cat)
            result['state'] = motion_cls_preds
        return result

    # def com_forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):
    #     x_0, x_1, x_2, x_3, x_4 = self.encode(bevs)
    #     if self.layer == 4:
    #         size = (1, 512, 16, 16)
    #         x_4 = self.communication(
    #             x_4, trans_matrices, num_agent_tensor, size, batch_size)
    #     elif self.layer == 3:
    #         size = (1, 256, 32, 32)
    #         x_3 = self.communication(
    #             x_3, trans_matrices, num_agent_tensor, size, batch_size)
    #     elif self.layer == 2:
    #         size = (1, 128, 64, 64)
    #         x_2 = self.communication(
    #             x_2, trans_matrices, num_agent_tensor, size, batch_size)
    #     elif self.layer == 1:
    #         size = (1, 64, 128, 128)
    #         x_1 = self.communication(
    #             x_1, trans_matrices, num_agent_tensor, size, batch_size)
    #     elif self.layer == 0:
    #         size = (1, 32, 256, 256)
    #         x_0 = self.communication(
    #             x_0, trans_matrices, num_agent_tensor, size, batch_size)

    #     x = self.decode(x_0, x_1, x_2, x_3, x_4, batch_size)
    #     result = self.head_out(x)
    #     return result

    # def single_com_forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):
    #     x_0, x_1, x_2, x_3, x_4 = self.encode(bevs)
    #     if self.layer == 4:
    #         size = (1, 512, 16, 16)
    #         x_4 = torch.cat([self.single_communication(
    #             x_4, trans_matrices, num_agent_tensor, size, batch_size, agent_index=i)
    #             for i in range(5)],
    #             dim=1)
    #         x_4 = x_4.transpose(0, 1).reshape(-1, *x_4.shape[-3:])
    #         feat_fuse_mat = x_4
    #     elif self.layer == 3:
    #         size = (1, 256, 32, 32)
    #         tmp = torch.cat([self.single_communication(
    #             x_3, trans_matrices, num_agent_tensor, size, batch_size, agent_index=i)
    #             for i in range(5)],
    #             dim=1)
    #         x_3 = tmp.transpose(0, 1).reshape(-1, *tmp.shape[-3:])
    #         feat_fuse_mat = x_3
    #     elif self.layer == 2:
    #         size = (1, 128, 64, 64)
    #         x_2 = torch.cat([self.single_communication(
    #             x_2, trans_matrices, num_agent_tensor, size, batch_size, agent_index=i)
    #             for i in range(5)],
    #             dim=1)
    #         x_2 = x_2.transpose(0, 1).reshape(-1, *x_2.shape[-3:])
    #         feat_fuse_mat = x_2
    #     elif self.layer == 1:
    #         size = (1, 64, 128, 128)
    #         x_1 = torch.cat([self.single_communication(
    #             x_1, trans_matrices, num_agent_tensor, size, batch_size, agent_index=i)
    #             for i in range(5)],
    #             dim=1)
    #         x_1 = x_1.transpose(0, 1).reshape(-1, *x_1.shape[-3:])
    #         feat_fuse_mat = x_1
    #     elif self.layer == 0:
    #         size = (1, 32, 256, 256)
    #         x_0 = torch.cat([self.single_communication(
    #             x_0, trans_matrices, num_agent_tensor, size, batch_size, agent_index=i)
    #             for i in range(5)],
    #             dim=1)
    #         x_0 = x_0.transpose(0, 1).reshape(-1, *x_0.shape[-3:])
    #         feat_fuse_mat = x_0

    #     if self.kd_flag == 1:
    #         x, x_8, x_7, x_6, x_5 = self.decode(
    #             x_0, x_1, x_2, x_3, x_4, batch_size)
    #         result = self.head_out(x)
    #         return result, x_8, x_7, x_6, x_5, feat_fuse_mat
    #     else:
    #         x = self.decode(x_0, x_1, x_2, x_3, x_4, batch_size)
    #         result = self.head_out(x)
    #         return result

    # def attack_forward(self, bevs, attack, trans_matrices, num_agent_tensor, batch_size=1, attack_mode='others', com=True):
    #     x_0, x_1, x_2, x_3, x_4 = self.encode(bevs)

    #     # att (b, n, eva_n, c, h, w)
    #     if self.layer == 4:
    #         size = (1, 512, 16, 16)
    #         x_4 = torch.cat(
    #             [self.single_communication(
    #                 x_4 +
    #                 self.place_attack(
    #                     attack[:, i, ...], num_agent_tensor[:, i], target_index=i, mode=attack_mode),
    #                 trans_matrices,
    #                 num_agent_tensor,
    #                 size,
    #                 batch_size,
    #                 agent_index=i,
    #                 com=com)
    #                 for i in range(5)],
    #             dim=1)
    #         x_4 = x_4.transpose(0, 1).reshape(-1, *x_4.shape[-3:])
    #         feat_fuse_mat = x_4
    #     elif self.layer == 3:
    #         size = (1, 256, 32, 32)
    #         tmp = torch.cat([
    #             self.single_communication(
    #                 x_3 +
    #                 self.place_attack(
    #                     attack[:, i, ...], num_agent_tensor[:, i], target_index=i, mode=attack_mode),
    #                 trans_matrices,
    #                 num_agent_tensor,
    #                 size,
    #                 batch_size,
    #                 agent_index=i,
    #                 com=com)
    #             for i in range(5)],
    #             dim=1)
    #         x_3 = tmp.transpose(0, 1).reshape(-1, *tmp.shape[-3:])
    #         feat_fuse_mat = x_3
    #     elif self.layer == 2:
    #         size = (1, 128, 64, 64)
    #         x_2 = torch.cat([self.single_communication(
    #             x_2, trans_matrices, num_agent_tensor, size, batch_size, agent_index=i)
    #             for i in range(5)],
    #             dim=1)
    #         x_2 = x_2.transpose(0, 1).reshape(-1, *x_2.shape[-3:])
    #         feat_fuse_mat = x_2
    #     elif self.layer == 1:
    #         size = (1, 64, 128, 128)
    #         x_1 = torch.cat([self.single_communication(
    #             x_1, trans_matrices, num_agent_tensor, size, batch_size, agent_index=i)
    #             for i in range(5)],
    #             dim=1)
    #         x_1 = x_1.transpose(0, 1).reshape(-1, *x_1.shape[-3:])
    #         feat_fuse_mat = x_1
    #     elif self.layer == 0:
    #         size = (1, 32, 256, 256)
    #         x_0 = torch.cat([self.single_communication(
    #             x_0, trans_matrices, num_agent_tensor, size, batch_size, agent_index=i)
    #             for i in range(5)],
    #             dim=1)
    #         x_0 = x_0.transpose(0, 1).reshape(-1, *x_0.shape[-3:])
    #         feat_fuse_mat = x_0

    #     if self.kd_flag == 1:
    #         x, x_8, x_7, x_6, x_5 = self.decode(
    #             x_0, x_1, x_2, x_3, x_4, batch_size)
    #         result = self.head_out(x)
    #         return result, x_8, x_7, x_6, x_5, feat_fuse_mat
    #     else:
    #         x = self.decode(x_0, x_1, x_2, x_3, x_4, batch_size)
    #         result = self.head_out(x)
    #         return result

    def com_forward_v2(self, bevs: torch.Tensor,
                       trans_matrices: torch.Tensor,
                       com_src: torch.Tensor,
                       com_tgt: torch.Tensor,
                       attack: Optional[torch.Tensor] = None,
                       attack_src: Optional[torch.Tensor] = None,
                       attack_tgt: Optional[torch.Tensor] = None,
                       batch_size: int = 1):

        x_0, x_1, x_2, x_3, x_4 = self.encode(bevs)

        feat_maps, size = self.select_com_layer(x_0, x_1, x_2, x_3, x_4)

        if attack is None:
            fused_feat = self.communication_v2(
                feat_maps, trans_matrices, com_src, com_tgt, size, batch_size)

        elif isinstance(attack, torch.Tensor):
            perturb = self.place_attack_v2(
                attack, attack_src, attack_tgt, com_src, com_tgt)
            fused_feat = self.communication_attack(
                feat_maps, trans_matrices, com_src, com_tgt, size, perturb, batch_size)

        x = self.decode(*self.place_merged_feature_back(x_0,
                        x_1, x_2, x_3, x_4, fused_feat), batch_size)
        if self.kd_flag:
            result = self.head_out(x[0])
            return (result,) + x[1:] + (fused_feat, )
        else:
            result = self.head_out(x)
            return result

    def after_encode(self, feat_maps: List[torch.Tensor],
                     trans_matrices: torch.Tensor,
                     com_src: torch.Tensor,
                     com_tgt: torch.Tensor,
                     attack: Optional[torch.Tensor] = None,
                     attack_src: Optional[torch.Tensor] = None,
                     attack_tgt: Optional[torch.Tensor] = None,
                     batch_size: int = 1,
                     batch_anchors: Optional[torch.Tensor] = None,
                     nms: bool = False, ):
        """
            To avoid duplicated encoding.
        """
        x_0, x_1, x_2, x_3, x_4 = feat_maps
        feat_maps, size = self.select_com_layer(x_0, x_1, x_2, x_3, x_4)

        if attack is None:
            fused_feat = self.communication_v2(
                feat_maps, trans_matrices, com_src, com_tgt, size, batch_size)

        elif isinstance(attack, torch.Tensor):
            perturb = self.place_attack_v2(
                attack, attack_src, attack_tgt, com_src, com_tgt)
            fused_feat = self.communication_attack(
                feat_maps, trans_matrices, com_src, com_tgt, size, perturb, batch_size)

        x = self.decode(*self.place_merged_feature_back(x_0,
                        x_1, x_2, x_3, x_4, fused_feat), batch_size)
        results = self.head_out(x)

        if nms:
            assert batch_anchors is not None
            # k = sum([torch.nonzero(b).shape[0] != 0 for b in bevs])  # non-empty agent num
            k = len(torch.unique(com_tgt, dim=0))
            return results, self.post_process(results, batch_anchors, k)
        else:
            return results

    def multi_com_forward(self, bevs: torch.Tensor,
                          trans_matrices: torch.Tensor,
                          com_srcs: List[torch.Tensor],
                          com_tgts: List[torch.Tensor],
                          attack: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                          attack_src: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                          attack_tgt: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                          batch_size: int = 1):

        assert type(attack) == type(attack_src) == type(attack_tgt)

        x_0, x_1, x_2, x_3, x_4 = self.encode(bevs)

        feat_maps, size = self.select_com_layer(x_0, x_1, x_2, x_3, x_4)

        result_list = []
        if attack is None:
            for com_src, com_tgt in zip(com_srcs, com_tgts):
                fused_feat = self.communication_v2(
                    feat_maps, trans_matrices, com_src, com_tgt, size, batch_size)

                x = self.decode(*self.place_merged_feature_back(x_0,
                                x_1, x_2, x_3, x_4, fused_feat), batch_size)
                result = self.head_out(x)
                result_list.append(result)

        elif isinstance(attack, torch.Tensor):
            for com_src, com_tgt in zip(com_srcs, com_tgts):
                perturb = self.place_attack_v2(
                    attack, attack_src, attack_tgt, com_src, com_tgt)
                fused_feat = self.communication_attack(
                    feat_maps, trans_matrices, com_src, com_tgt, size, perturb, batch_size)

                x = self.decode(*self.place_merged_feature_back(x_0,
                                x_1, x_2, x_3, x_4, fused_feat), batch_size)
                result = self.head_out(x)
                result_list.append(result)

        elif isinstance(attack, List):
            for com_src, com_tgt, att_src, att_tgt in zip(com_srcs, com_tgts, attack_src, attack_tgt):
                perturb = self.place_attack_v2(
                    attack, att_src, att_tgt, com_src, com_tgt)
                fused_feat = self.communication_attack(
                    feat_maps, trans_matrices, com_src, com_tgt, size, perturb, batch_size)

                x = self.decode(*self.place_merged_feature_back(x_0,
                                x_1, x_2, x_3, x_4, fused_feat), batch_size)
                result = self.head_out(x)
                result_list.append(result)

        return result_list

    def single_view_multi_com_forward(self, bevs: torch.Tensor,
                                      trans_matrices: torch.Tensor,
                                      com_srcs: List[torch.Tensor],
                                      com_tgts: List[torch.Tensor],
                                      attack: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                                      attack_src: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                                    #   attack_tgt: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                                      batch_size: int = 1):
        """
            单个agent视角下，用接收到的其他agent的信息，模拟其他agent的inference过程
        """

        assert type(attack) == type(attack_src)
        # assert (attack_tgt[:, 1] == attack_tgt[0, 1]).all()

        x_0, x_1, x_2, x_3, x_4 = self.encode(bevs)

        feat_maps, size = self.select_com_layer(x_0, x_1, x_2, x_3, x_4)
        # import ipdb;ipdb.set_trace()
        if isinstance(attack, torch.Tensor):
            for (b, agent_id), att in zip(attack_src, attack):
                feat_maps[agent_id] += att

        result_list = []
        for com_src, com_tgt in zip(com_srcs, com_tgts):
            fused_feat = self.communication_v2(
                feat_maps, trans_matrices, com_src, com_tgt, size, batch_size)

            x = self.decode(*self.place_merged_feature_back(x_0,
                            x_1, x_2, x_3, x_4, fused_feat), batch_size)
            result = self.head_out(x)
            result_list.append(result)

        return result_list

    # def place_attack(self, attack: torch.Tensor, num_agent_tensor: torch.Tensor, target_index: int, mode='others'):
    #     # attack: (b, eva_n, c, h, w)
    #     b, eva_num, c, h, w = attack.shape
    #     assert eva_num < 5
    #     assert mode in ['others', 'self']
    #     template = torch.zeros(
    #         (b, 5, c, h, w), dtype=attack.dtype, device=attack.device)
    #     for i in range(b):
    #         if num_agent_tensor[i] == 0 or num_agent_tensor[i] <= eva_num:
    #             continue
    #         if mode == 'others':
    #             place_index = (torch.arange(eva_num).cuda() +
    #                            target_index + 1) % num_agent_tensor[i]
    #         elif mode == 'self':
    #             assert eva_num == 1
    #             place_index = target_index
    #         template[i, place_index, ...] = attack[i]

    #     placed_attack = self.agents2batch(template)
    #     return placed_attack

    def place_attack_v2(self, attack: torch.Tensor,
                        attack_src: torch.Tensor,
                        attack_tgt: torch.Tensor,
                        com_src: torch.Tensor,
                        com_tgt: torch.Tensor):
        lpt = dict()
        for i, (src, tgt) in enumerate(zip(com_src, com_tgt)):
            lpt[(src[0].item(), src[1].item(), tgt[1].item())] = i

        index = torch.zeros(
            com_src.shape[0], dtype=torch.long, device=attack.device)
        for j, (src, tgt) in enumerate(zip(attack_src, attack_tgt)):
            if (src[0].item(), src[1].item(), tgt[1].item()) in lpt:
                index[lpt[(src[0].item(), src[1].item(), tgt[1].item())]] = j+1

        padded_attack = torch.cat(
            [attack.new_zeros(1, *attack.shape[1:]), attack], dim=0)

        return padded_attack[index]

    def get_default_com_pair(self, num_agent_tensor: torch.Tensor):
        com_src = []
        com_tgt = []
        for b in range(len(num_agent_tensor)):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                com_src.append([b, i])
                com_tgt.append([b, i])
                for j in range(num_agent):
                    if j != i:
                        com_src.append([b, j])
                        com_tgt.append([b, i])
        com_src = torch.Tensor(com_src).long().to(num_agent_tensor.device)
        com_tgt = torch.Tensor(com_tgt).long().to(num_agent_tensor.device)

        return com_src, com_tgt

    def get_com_pair(self, num_agent_tensor: torch.Tensor, n_com: Optional[int] = None):
        """
            args:
                num_agent_tensor: [b, 5]
                n_com: number of maximum communication agents, `None` means all possible and `1` means no communication
        """
        if n_com is None:
            return self.get_default_com_pair(num_agent_tensor)
        else:
            com_src = []
            com_tgt = []
            for b, num_agent in enumerate(num_agent_tensor):
                for i, n in enumerate(num_agent):
                    # min(n_com, n) can filter out the padded agents (n=0)
                    for j in range(min(n_com, n.item())):
                        com_src.append([b, (i+j) % n.item()])
                        com_tgt.append([b, i])
            com_src = torch.Tensor(com_src).long().to(num_agent_tensor.device)
            com_tgt = torch.Tensor(com_tgt).long().to(num_agent_tensor.device)
            return com_src, com_tgt

    def get_attack_pair(self, num_agent_tensor: torch.Tensor,
                        n_com: Optional[int] = None,
                        n_att: int = 1,
                        mode: str = 'others'):
        att_src = []
        att_tgt = []
        for b, num_agent in enumerate(num_agent_tensor):
            for i, n in enumerate(num_agent):
                # assert n > n_att, f"num_agent {n} should be greater than n_att {n_att}"
                if mode == 'others' and n <= n_att:
                    continue
                base = min(n.item(), n_com) if n_com is not None else n.item()
                for j in range(n_att):
                    if mode == 'others':
                        att_src.append([b, (i+j+1) % base])
                        att_tgt.append([b, i])
                    elif mode == 'self':
                        assert n_att == 1
                        att_src.append([b, i])
                        att_tgt.append([b, i])

        att_src = torch.Tensor(att_src).long().to(num_agent_tensor.device)
        att_tgt = torch.Tensor(att_tgt).long().to(num_agent_tensor.device)
        return att_src, att_tgt

    def get_attack_det_com_pairs(self, num_agent_tensor: torch.Tensor):
        """
            only support batch_size=1
        """
        assert num_agent_tensor.shape[0] == 1, "only support batch_size=1"
        ret_list = []
        num_agents = num_agent_tensor[0, 0].item()  # n communication pairs

        # no communication
        com_src = []
        com_tgt = []
        for i, n in enumerate(num_agent_tensor[0]):
            if n > 0:
                com_src.append([0, i])
                com_tgt.append([0, i])
        com_src = torch.Tensor(com_src).long().to(num_agent_tensor.device)
        com_tgt = torch.Tensor(com_tgt).long().to(num_agent_tensor.device)
        ret_list.append((com_src, com_tgt))

        for offset in range(1, num_agents):
            com_src = []
            com_tgt = []
            for i, n in enumerate(num_agent_tensor[0]):
                if n > 0:
                    com_src.append([0, i])
                    com_tgt.append([0, i])

                    com_src.append([0, (i+offset) % n.item()])
                    com_tgt.append([0, i])
            com_src = torch.Tensor(com_src).long().to(num_agent_tensor.device)
            com_tgt = torch.Tensor(com_tgt).long().to(num_agent_tensor.device)
            ret_list.append((com_src, com_tgt))

        com_srcs, com_tgts = zip(*ret_list)
        return list(com_srcs), list(com_tgts)

    def get_attack_det_com_pairs_rm1(self, num_agent_tensor: torch.Tensor):
        """
            only support batch_size=1
        """
        assert num_agent_tensor.shape[0] == 1, "only support batch_size=1"
        ret_list = []
        num_agents = num_agent_tensor[0, 0].item()  # n communication pairs
        
        com_src_def, com_tgt_def = self.get_default_com_pair(num_agent_tensor)
        ret_list.append((com_src_def.clone(), com_tgt_def.clone()))

        srcs_det, tgts_det = [], []
        for offset in range(1, num_agents):
            com_src, com_tgt = com_src_def.clone(), com_tgt_def.clone()
            src_det, tgt_det = [], []
            for i, n in enumerate(num_agent_tensor[0]):
                if n > 0:
                    src_det.append([0, (i+offset) % n.item()])
                    tgt_det.append([0, i])
            src_det = torch.Tensor(src_det).long().to(num_agent_tensor.device)
            tgt_det = torch.Tensor(tgt_det).long().to(num_agent_tensor.device)
            ret_list.append(self.remove_com_pair(com_src, com_tgt, src_det, tgt_det)) 
            srcs_det.append(src_det)
            tgts_det.append(tgt_det)
        
        com_srcs, com_tgts = zip(*ret_list)
        return list(com_srcs), list(com_tgts), torch.cat(srcs_det), torch.cat(tgts_det)


    def remove_com_pair(self, com_src, com_tgt, rm_src, rm_tgt):
        lpt = set()
        for src, tgt in zip(rm_src, rm_tgt):
            lpt.add((src[0].item(), src[1].item(), tgt[1].item()))
        
        new_com_src = []
        new_com_tgt = []
        for src, tgt in zip(com_src, com_tgt):
            if (src[0].item(), src[1].item(), tgt[1].item()) not in lpt:
                new_com_src.append(src)
                new_com_tgt.append(tgt)
        
        new_com_src = torch.stack(new_com_src, dim=0)
        new_com_tgt = torch.stack(new_com_tgt, dim=0)

        return new_com_src, new_com_tgt

    def post_process(self, results, batch_anchors, k):
        """
            Inputs: box_preds, cls_preds, anchors, k(nonempty)
            Outputs: box predictions (after nms)
        """

        predictions_dicts = []

        batch_box_preds = results['loc'][:k]  # (b, h, w, 6, 1, box_dim)
        batch_cls_preds = results['cls'][:k]  # (b, h*w*6*1, c)
        batch_anchors = batch_anchors[:k]

        batch_box_preds = batch_box_preds.reshape(
            batch_box_preds.shape[0], -1, batch_box_preds.shape[-2], batch_box_preds.shape[-1])
        batch_anchors = batch_anchors.reshape(
            batch_anchors.shape[0], -1, batch_anchors.shape[-1])

        for box_preds, cls_preds, anchors in zip(
                batch_box_preds, batch_cls_preds, batch_anchors):
            total_scores = F.softmax(
                cls_preds, dim=-1)[..., 1:]  # ignore background
            decoded_boxes_torch = bev_box_decode_torch(
                box_preds[:, 0], anchors)
            decoded_boxes_torch = rescale_boxes(decoded_boxes_torch)
            rotated_box = torch.cat([decoded_boxes_torch[..., :4], sincos2deg(
                decoded_boxes_torch[..., 4:])], dim=-1)

            class_selected = []
            for i in range(total_scores.shape[1]):
                scores = total_scores[:, i]
                filt_id = torch.where(scores > 0.7)[0]
                if filt_id.shape[0] > 0:
                    selected_idx = nms_rotated(
                        rotated_box[filt_id], scores[filt_id], iou_threshold=0.01)
                    selected_idx = filt_id[selected_idx].detach().cpu().numpy()
                else:
                    # no object detected
                    selected_idx = filt_id.detach().cpu().numpy()

                box_corners = center_to_corner_box2d_torch(
                    decoded_boxes_torch[selected_idx, :2],
                    decoded_boxes_torch[selected_idx, 2:4],
                    decoded_boxes_torch[selected_idx, 4:]).unsqueeze(1)

                class_selected.append({
                    'pred': box_corners.detach().cpu().numpy(),                      # np.ndarray
                    # np.ndarray
                    'score': total_scores[selected_idx, i].detach().cpu().numpy(),
                    # torch.Tensor
                    'rot_box': rotated_box[selected_idx],
                    'selected_idx': selected_idx,                                   # np.ndarray
                })

            predictions_dicts.append(class_selected)

        return predictions_dicts  # , cls_pred_first_nms

    def debug_com(self, bevs: torch.Tensor,
                  trans_matrices: torch.Tensor,
                  attack1: torch.Tensor,
                  num_agent_tensor: torch.Tensor,
                  com_src: torch.Tensor,
                  com_tgt: torch.Tensor,
                  attack2: Optional[torch.Tensor] = None,
                  attack_src: Optional[torch.Tensor] = None,
                  attack_tgt: Optional[torch.Tensor] = None,
                  batch_size: int = 1):
        x_0, x_1, x_2, x_3, x_4 = self.encode(bevs)

        feat_maps, size = self.select_com_layer(x_0, x_1, x_2, x_3, x_4)

        attacked_feats_list1 = [x_3 + self.place_attack(attack1[:, i, ...],
                                                        num_agent_tensor[:, i], target_index=i, mode='others') for i in range(5)]

        tmp = torch.cat([self.single_communication(f, trans_matrices, num_agent_tensor,
                        size, batch_size, agent_index=i) for i, f in enumerate(attacked_feats_list1)])
        fused_feat_maps1 = tmp.transpose(0, 1).reshape(-1, *tmp.shape[-3:])

        perturb = self.place_attack_v2(
            attack2, attack_src, attack_tgt, com_src, com_tgt)
        fused_feat = self.communication_attack(
            feat_maps, trans_matrices, com_src, com_tgt, size, perturb, batch_size)


class V2VNet(nn.Module):
    def __init__(self, config, gnn_iter_times, layer, layer_channel, in_channels=13):
        super(V2VNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.layer = layer
        self.layer_channel = layer_channel

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)

        self.gnn_iter_num = gnn_iter_times
        self.convgru = convrnn.Conv2dGRU(in_channels=self.layer_channel * 2,
                                         out_channels=self.layer_channel,
                                         kernel_size=3,
                                         num_layers=1,
                                         bidirectional=False,
                                         dilation=1,
                                         stride=1)

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):
        # trans_matrices [batch 5 5 4 4]
        # num_agent_tensor, shape: [batch, num_agent]; how many non-empty agent in this scene

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x_0, x_1, x_2, x_3, x_4 = self.u_encoder(bevs)
        device = bevs.device

        if self.layer == 4:
            feat_maps = x_4
            size = (1, 512, 16, 16)
        elif self.layer == 3:
            feat_maps = x_3
            size = (1, 256, 32, 32)
        elif self.layer == 2:
            feat_maps = x_2
            size = (1, 128, 64, 64)
        elif self.layer == 1:
            feat_maps = x_1
            size = (1, 64, 128, 128)
        elif self.layer == 0:
            feat_maps = x_0
            size = (1, 32, 256, 256)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}

        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(
                feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat = torch.cat(tuple(feat_list), 1)

        # to avoid the inplace operation
        local_com_mat_update = torch.cat(tuple(feat_list), 1)
        p = np.array([1.0, 0.0])

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]

            agent_feat_list = list()
            for nb in range(self.agent_num):  # self.agent_num = 5
                agent_feat_list.append(local_com_mat[b, nb])

            for _ in range(self.gnn_iter_num):

                updated_feats_list = list()

                for i in range(num_agent):
                    neighbor_feat_list = list()
                    # transformation [2 5 5 4 4]
                    all_warp = trans_matrices[b, i]
                    com_outage = np.random.choice([0, 1], p=p.ravel())

                    if com_outage == 0:
                        for j in range(num_agent):
                            if j != i:
                                nb_agent = torch.unsqueeze(
                                    agent_feat_list[j], 0)  # [1 512 16 16]
                                nb_warp = all_warp[j]  # [4 4]
                                # normalize the translation vector
                                x_trans = (4*nb_warp[0, 3])/128
                                y_trans = -(4*nb_warp[1, 3])/128

                                theta_rot = torch.tensor([[nb_warp[0, 0], nb_warp[0, 1], 0.0], [
                                                         nb_warp[1, 0], nb_warp[1, 1], 0.0]]).type(dtype=torch.float).to(device)
                                theta_rot = torch.unsqueeze(theta_rot, 0)
                                # 得到grid 用于grid sample
                                grid_rot = F.affine_grid(
                                    theta_rot, size=torch.Size(size))

                                theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(
                                    dtype=torch.float).to(device)
                                theta_trans = torch.unsqueeze(theta_trans, 0)
                                grid_trans = F.affine_grid(
                                    theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                                # first rotate the feature map, then translate it
                                warp_feat_rot = F.grid_sample(
                                    nb_agent, grid_rot, mode='bilinear')
                                warp_feat_trans = F.grid_sample(
                                    warp_feat_rot, grid_trans, mode='bilinear')
                                warp_feat = torch.squeeze(warp_feat_trans)

                                neighbor_feat_list.append(warp_feat)

                        mean_feat = torch.mean(torch.stack(
                            neighbor_feat_list), dim=0)  # [c, h, w]
                        cat_feat = torch.cat(
                            [agent_feat_list[i], mean_feat], dim=0)
                        cat_feat = cat_feat.unsqueeze(
                            0).unsqueeze(0)  # [1, 1, c, h, w]
                        updated_feat, _ = self.convgru(cat_feat, None)
                        updated_feat = torch.squeeze(
                            torch.squeeze(updated_feat, 0), 0)  # [c, h, w]
                        updated_feats_list.append(updated_feat)

                    else:
                        updated_feats_list.append(agent_feat_list[i])

                agent_feat_list = updated_feats_list

            for k in range(num_agent):
                local_com_mat_update[b, k] = agent_feat_list[k]

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)

        if self.layer == 4:
            x = self.decoder(x_0, x_1, x_2, x_3, feat_fuse_mat, batch_size)
        elif self.layer == 3:
            x = self.decoder(x_0, x_1, x_2, feat_fuse_mat, x_4, batch_size)
        elif self.layer == 2:
            x = self.decoder(x_0, x_1, feat_fuse_mat, x_3, x_4, batch_size)
        elif self.layer == 1:
            x = self.decoder(x_0, feat_fuse_mat, x_2, x_3, x_4, batch_size)
        elif self.layer == 0:
            x = self.decoder(feat_fuse_mat, x_1, x_2, x_3, x_4, batch_size)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0], -1, self.category_num)

        # Detection head
        loc_preds = self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1, loc_preds.size(1), loc_preds.size(
            2), self.anchor_num_per_loc, self.out_seq_len, self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        return result


class FaFNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError


class When2com(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError


class SumFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError


class MeanFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError


class MaxFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError


class CatFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError


class AgentwiseWeightedFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError


class AgentWeightedFusion(nn.Module):
    def __init__(self, config):
        super(AgentWeightedFusion, self).__init__()

        self.conv1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

        # self.conv1_1 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_1 = nn.BatchNorm2d(1)
        self.conv1_5 = nn.Conv2d(1, 1, kernel_size=32, stride=1, padding=0)
        # # self.bn1_2 = nn.BatchNorm2d(1)

    def forward(self, x):
        # x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        # x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        # x_1 = F.sigmoid(self.conv1_2(x_1))
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))
        x_1 = F.relu(self.conv1_5(x_1))

        return x_1


class ClassificationHead(nn.Module):

    def __init__(self, config):

        super(ClassificationHead, self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(
            channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            channel, category_num*anchor_num_per_loc, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class SingleRegressionHead(nn.Module):
    def __init__(self, config):
        super(SingleRegressionHead, self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size
        if config.only_det:
            out_seq_len = 1
        else:
            out_seq_len = config.pred_len

        if config.binary:
            if config.only_det:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3,
                              stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(channel, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1, padding=0))
            else:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, 128, kernel_size=3,
                              stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        box = self.box_prediction(x)

        return box


class ModulationLayer3(nn.Module):
    def __init__(self, config):
        super(ModulationLayer3, self).__init__()

        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))

        return x_1


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(
            channel*2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1
