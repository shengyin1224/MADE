# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# a class that integrate multiple simple fusion methods (Single Scale)
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from collections import OrderedDict
import torch
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.feature_alignnet import AlignNet, DeformAlignNet
from opencood.models.sub_modules.refactor import refactor
from opencood.models.sub_modules.view_embedding import ViewEmbedding, ViewEmbeddingIdentity
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion, warp_feature
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.second_ssfa import SecondSSFA
from opencood.models.lift_splat_shoot import LiftSplatShoot
from opencood.models.sub_modules.matcher_v3 import MatcherV3
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import FpvrcnnPostprocessor
from opencood.models.sub_modules.bev_roi_head import BEVRoIHead


class HeterSECONDLiftSplat2Stage(nn.Module):
    def __init__(self, args):
        super(HeterSECONDLiftSplat2Stage, self).__init__()
        self.lidar_encoder = SecondSSFA(args['lidar_args'])
        self.camera_encoder = LiftSplatShoot(args['camera_args'])
        self.cav_range = args['lidar_args']['lidar_range']
        self.max_cav = args['lidar_args']['max_cav']

        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        """
        Transformer encoder fusion
        input require (seq, batch, feature)
        """
        # self.fuse_module = nn.TransformerEncoderLayer(d_model=args['fuse_args']['feat_dim'], # 128
        #                                               nhead=args['fuse_args']['nhead'], # 4
        #                                               dim_feedforward=args['fuse_args']['ffn_dim']) # 128
        # self.fuse_net = nn.TransformerEncoder(encoder_layer=self.fuse_module, 
        #                                       num_layers=args['fuse_args']['nlayer'])

        """
        Shrink head for unify feature channels
        """
        self.shrink_lidar = DownsampleConv(args['shrink_header_lidar'])
        self.shrink_camera = DownsampleConv(args['shrink_header_camera'])

        """
        Domain alignment net for lidar and camera
        """
        self.lidar_aligner = AlignNet(args.get("lidar_aligner", "SDTA"))
        self.camera_aligner = DeformAlignNet(args.get("lidar_aligner", "SDTA"))

        """
        1 stage Heads
        """
        self.cls_head_lidar = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head_lidar = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head_lidar = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        self.iou_head_lidar = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1, bias=False)

        self.cls_head_camera = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head_camera = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head_camera = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        self.iou_head_camera = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1, bias=False)

        """
        Shared Heads
        """
        self.cls_head_shared = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head_shared = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head_shared = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        self.iou_head_shared = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1, bias=False)

        """
        1 stage postprocessor
        """
        self.post_processor = FpvrcnnPostprocessor(args['post_processer'],
                                                   train=self.training)
        self.matcher = MatcherV3(args['matcher'], self.cav_range)
        self.view_embedding = ViewEmbeddingIdentity() if args['view_embedding'] == 'identity' else \
                              ViewEmbedding(args['view_embedding'])
        self.roi_head = BEVRoIHead(args['bev_roi_head'], self.cav_range)
        self.train_stage2 = args['activate_stage2']

        if 'freeze_lidar' in args and args['freeze_lidar']:
            self.freeze_lidar()
        if 'freeze_camera' in args and args['freeze_camera']:
            self.freeze_camera()

    def freeze_lidar(self):
        for p in self.lidar_encoder.parameters():
            p.requires_grad_(False)
        for p in self.shrink_lidar.parameters():
            p.requires_grad_(False)
        for p in self.cls_head_lidar.parameters():
            p.requires_grad_(False)
        for p in self.reg_head_lidar.parameters():
            p.requires_grad_(False)
        for p in self.dir_head_lidar.parameters():
            p.requires_grad_(False)
        for p in self.iou_head_lidar.parameters():
            p.requires_grad_(False)

    def freeze_camera(self):
        for p in self.camera_encoder.parameters():
            p.requires_grad_(False)
        for p in self.shrink_camera.parameters():
            p.requires_grad_(False)
        for p in self.cls_head_camera.parameters():
            p.requires_grad_(False)
        for p in self.reg_head_camera.parameters():
            p.requires_grad_(False)
        for p in self.dir_head_camera.parameters():
            p.requires_grad_(False)
        for p in self.iou_head_camera.parameters():
            p.requires_grad_(False)
 

    def forward(self, batch_dict):
        output_dict = OrderedDict()

        lidar_agent_indicator = batch_dict['lidar_agent_record'] # [sum(record_len)]
        print(lidar_agent_indicator)
        record_len = batch_dict['record_len']

        skip_lidar, skip_camera = False, False
        if sum(lidar_agent_indicator) == sum(record_len):
            skip_camera = True
        if sum(lidar_agent_indicator) == 0:
            skip_lidar = True

        # LiDAR Encode
        if not skip_lidar:
            voxel_features = batch_dict['processed_lidar']['voxel_features']
            voxel_coords = batch_dict['processed_lidar']['voxel_coords']
            voxel_num_points = batch_dict['processed_lidar']['voxel_num_points']

            batch_dict.update({'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'batch_size': sum(lidar_agent_indicator)})

            batch_dict = self.lidar_encoder.vfe(batch_dict)
            batch_dict = self.lidar_encoder.spconv_block(batch_dict)
            batch_dict = self.lidar_encoder.map_to_bev(batch_dict)
            lidar_feature_2d = self.lidar_encoder.ssfa(batch_dict['spatial_features'])
            lidar_feature_2d = self.shrink_lidar(lidar_feature_2d) 

            # 1-stage decode
            cls_preds_lidar = self.cls_head_lidar(lidar_feature_2d)
            reg_preds_lidar = self.reg_head_lidar(lidar_feature_2d)
            dir_preds_lidar = self.dir_head_lidar(lidar_feature_2d)
            iou_preds_lidar = self.iou_head_lidar(lidar_feature_2d)

            lidar_feature_2d = self.lidar_aligner(lidar_feature_2d)

        # Camera Encode
        if not skip_camera:
            image_inputs_dict = batch_dict['image_inputs']
            x, rots, trans, intrins, post_rots, post_trans = \
                image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
            x, depth_items = self.camera_encoder.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)
            camera_feature_2d = self.camera_encoder.bevencode(x) 
            camera_feature_2d = self.shrink_camera(camera_feature_2d)

            # 1-stage decode
            cls_preds_camera = self.cls_head_camera(camera_feature_2d)
            reg_preds_camera = self.reg_head_camera(camera_feature_2d)
            dir_preds_camera = self.dir_head_camera(camera_feature_2d)
            iou_preds_camera = self.iou_head_camera(camera_feature_2d)
            
            # feature align.
            camera_feature_2d = self.camera_aligner(camera_feature_2d)

        # stage1 output
        batch_dict['stage1_out'] = {}
        if skip_camera:
            heter_feature_2d = lidar_feature_2d
            for branch in ['cls_preds', 'reg_preds', 'dir_preds']: #, 'iou_preds']:
                batch_dict['stage1_out'].update({branch: eval(f"{branch}_lidar")})
            
        elif skip_lidar:
            heter_feature_2d = camera_feature_2d
            for branch in ['cls_preds', 'reg_preds', 'dir_preds']: #, 'iou_preds']:
                batch_dict['stage1_out'].update({branch: eval(f"{branch}_camera")})

        else:
            batch_dict['stage1_out']['cls_preds'] = []
            batch_dict['stage1_out']['reg_preds'] = []
            batch_dict['stage1_out']['dir_preds'] = []
            # batch_dict['stage1_out']['iou_preds'] = []
            heter_feature_2d = []

            camera_idx = 0
            lidar_idx = 0

            for i in range(sum(record_len)): 
                if lidar_agent_indicator[i]:
                    heter_feature_2d.append(lidar_feature_2d[lidar_idx])
                    for branch in ['cls_preds', 'reg_preds', 'dir_preds']: #, 'iou_preds']:
                        batch_dict['stage1_out'][branch].append(eval(f"{branch}_lidar")[lidar_idx])
                    lidar_idx += 1
                else:
                    heter_feature_2d.append(camera_feature_2d[camera_idx])
                    for branch in ['cls_preds', 'reg_preds', 'dir_preds']: #, 'iou_preds']:
                        batch_dict['stage1_out'][branch].append(eval(f"{branch}_camera")[camera_idx])
                    camera_idx += 1
            
            heter_feature_2d = torch.stack(heter_feature_2d)
            for branch in ['cls_preds', 'reg_preds', 'dir_preds']: #, 'iou_preds']:
                batch_dict['stage1_out'][branch] = torch.stack(batch_dict['stage1_out'][branch])


        # shared head
        batch_dict['shared_head_out'] = OrderedDict()
        cls_preds_shared = self.cls_head_shared(heter_feature_2d)
        reg_preds_shared = self.reg_head_shared(heter_feature_2d)
        dir_preds_shared = self.dir_head_shared(heter_feature_2d)
        # iou_preds_shared = self.iou_head_shared(heter_feature_2d)
        for branch in ['cls_preds', 'reg_preds', 'dir_preds']:
            batch_dict['shared_head_out'].update({branch: eval(f"{branch}_shared")})

        # # use aligned feature to decode stage1 box
        # for branch in ['cls_preds', 'reg_preds', 'dir_preds']:
        #     batch_dict['stage1_out'][branch] = eval(f"{branch}_shared")

        # stage1 decode
        data_dict, output_dict = {}, {}
        data_dict['ego'], output_dict['ego'] = batch_dict, batch_dict

        pred_box3d_list, scores_list = \
            self.post_processor.post_process(data_dict, output_dict,
                                             stage1=True)

        batch_dict.pop('stage1_out')

        # save stage1 result          
        batch_dict['det_boxes'] = pred_box3d_list
        batch_dict['det_scores'] = scores_list
        
        if pred_box3d_list is not None and self.train_stage2:
            t_matrix = normalize_pairwise_tfm(batch_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
            
            heter_feature_2d = warp_feature(heter_feature_2d, record_len, t_matrix) # [N, C, H, W]

            batch_dict['feature_shape'] = heter_feature_2d.shape[2:] # [H, W]
            batch_dict['batch_size_stage2'] = len(record_len)
            batch_dict = self.matcher(batch_dict) # 'boxes_fused', 'scores_fused', 'agentid_fused', 'roi_fused'
            if torch.cat(batch_dict['boxes_fused'], dim=0).shape[0] == 0:
                return batch_dict
            heter_feature_2d = self.view_embedding(heter_feature_2d, batch_dict) 

            batch_dict = refactor(batch_dict, lidar_agent_indicator) # 'lidar_matrix_list', 'camera_matrix_list'
            
            lidar_matrix = torch.cat(batch_dict['lidar_matrix_list'], dim=0) # [sum(proposal_num), sum(agent_num)]
            camera_matrix = torch.cat(batch_dict['camera_matrix_list'], dim=0) # [sum(proposal_num), sum(agent_num)]
            roi_fused_cat = torch.cat(batch_dict['roi_fused'], dim=0) # [sum(proposal_num), 4]

            lidar_roi_agg_list = []
            camera_roi_agg_list = []

            agentnum_of_proposals = []
            pixelnum_of_proposals = []
            mask_of_proposals = []
            random_cav_of_proposals = [] # For fused kd loss
            feature_of_proposals = []

            # roi_fused_cat [sum(proposal_num), 4]
            for i, roi in enumerate(roi_fused_cat):
                # roi is [xmin, xmax, ymin, ymax]

                # if two modality exist, perform knowledge distillation
                if sum(lidar_matrix[i]) != 0 and sum(camera_matrix[i]) != 0:
                    if heter_feature_2d[camera_matrix[i]==1].shape[0] == 0 or heter_feature_2d[lidar_matrix[i]==1].shape[0] == 0:
                        return batch_dict # strange bug !

                    lidar_roi_agg = heter_feature_2d[lidar_matrix[i]==1].max(dim=0)[0][..., roi[2]:roi[3], roi[0]:roi[1]]
                    camera_roi_agg = heter_feature_2d[camera_matrix[i]==1].max(dim=0)[0][..., roi[2]:roi[3], roi[0]:roi[1]]

                    # [C, RoI_H*RoI_W]
                    lidar_roi_agg_list.append(lidar_roi_agg.flatten(start_dim=1))
                    camera_roi_agg_list.append(camera_roi_agg.flatten(start_dim=1))

                # [N_source, C, RoI_H, RoI_W]
                all_roi_agg = heter_feature_2d[torch.logical_or(lidar_matrix[i]==1, camera_matrix[i]==1)][..., roi[2]:roi[3], roi[0]:roi[1]]

                agentnum = all_roi_agg.shape[0]
                pixelnum = all_roi_agg.shape[2]*all_roi_agg.shape[3]

                agentnum_of_proposals.append(agentnum)
                pixelnum_of_proposals.append(pixelnum)

                # [RoI_Hi*RoI_Wi, max_cav]
                mask_of_proposals.append(
                        torch.tensor([0]*agentnum + [1]*(self.max_cav-agentnum)).expand(pixelnum, self.max_cav)
                    )
                
                random_cav = torch.zeros((self.max_cav), dtype=torch.long)
                if agentnum > 1:
                    random_cav[np.random.randint(1, agentnum)] = 1
                else:
                    random_cav[0] = 1
                random_cav_of_proposals.append(
                    random_cav.expand(pixelnum, self.max_cav)
                )

                # [max_cav, C, RoI_H*RoI_W]
                feature_of_proposals.append(F.pad(all_roi_agg.flatten(start_dim=2), (0,0,0,0,0,self.max_cav-all_roi_agg.shape[0]), mode='constant', value=0))


            # [max_cav, sum(RoI_Hi*RoI_Wi), C].     (seq, batch, feature)
            feature_of_proposals = torch.cat(feature_of_proposals, dim=-1).permute(0,2,1)

            """
            # binary mask. [sum(RoI_Hi*RoI_Wi), max_cav]. (batch, seq) a BoolTensor is provided, the positions with the value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            src_key_padding_mask = torch.cat(mask_of_proposals, dim=0).to(feature_of_proposals.device)

            # [max_cav, sum(RoI_Hi*RoI_Wi), C]
            # feature_of_proposals = self.fuse_net(feature_of_proposals, src_key_padding_mask=src_key_padding_mask)

            # recover to each proposal
            # [sum(RoI_Hi*RoI_Wi), C]
            feature_of_proposals_ego = feature_of_proposals[0, ...]
            """

            feature_of_proposals_ego = feature_of_proposals.max(dim=0)[0]

            # list of flatten proposal feature: [[RoI_H0*RoI_W0, C], [RoI_H1*RoI_W1, C], ...]
            feature_of_proposals_ego_list = torch.split(feature_of_proposals_ego, pixelnum_of_proposals, dim=0)
            
            batch_dict['feature_of_proposals_ego_list'] = feature_of_proposals_ego_list
            batch_dict['batch_size_2stage'] = len(record_len)
            batch_dict = self.roi_head(batch_dict)

            # Knowledge distillation. L2 Loss
            if len(lidar_roi_agg_list) != 0:
                batch_dict['kd_items'] = OrderedDict()

                batch_dict['kd_items']["lidar_roi_features"] = torch.cat(lidar_roi_agg_list, dim=1)
                batch_dict['kd_items']["camera_roi_features"] = torch.cat(camera_roi_agg_list, dim=1)
            
            # Transformer consistency fusion supervision.

            # random_cav_of_proposals_cat = torch.cat(random_cav_of_proposals, dim=0)
            # if not torch.all(random_cav_of_proposals_cat[:, 0]):
            #     batch_dict['cons_items'] = OrderedDict()
            #     batch_dict['cons_items']["random_cav_mask"] = random_cav_of_proposals_cat # [sum(RoI_Hi*RoI_Wi), max_cav]
            #     # [max_cav, sum(RoI_Hi*RoI_Wi), C] -> [sum(RoI_Hi*RoI_Wi), max_cav, C]
            #     batch_dict['cons_items']["fused_roi_feature"] = feature_of_proposals.permute(1,0,2)  # [sum(RoI_Hi*RoI_Wi), max_cav, C]

        return batch_dict
