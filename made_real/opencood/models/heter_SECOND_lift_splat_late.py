# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# a class that integrate multiple simple fusion methods (Single Scale)
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm

import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.second_ssfa import SecondSSFA
from opencood.models.lift_splat_shoot import LiftSplatShoot

class HeterSECONDLiftSplatLate(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(HeterSECONDLiftSplatLate, self).__init__()
        self.lidar_encoder = SecondSSFA(args['lidar_args'])
        self.camera_encoder = LiftSplatShoot(args['camera_args'])
        
        """
        Shrink head for unify feature channels
        """
        self.shrink_lidar = DownsampleConv(args['shrink_header_lidar'])
        self.shrink_camera = DownsampleConv(args['shrink_header_camera'])


        """
        Shared Heads
        """
        self.cls_head_lidar = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head_lidar = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head_lidar = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        self.cls_head_camera = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head_camera = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head_camera = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        
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
 

    def forward(self, data_dict):
        if 'image_inputs' in data_dict:
            """
            Camera Encode
            """
            image_inputs_dict = data_dict['image_inputs']
            x, rots, trans, intrins, post_rots, post_trans = \
                image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
            x, depth_items = self.camera_encoder.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)
            camera_feature_2d = self.camera_encoder.bevencode(x)  # H0, W0
            camera_feature_2d = self.shrink_camera(camera_feature_2d) # H0/2, W0/2
            feature = camera_feature_2d

            cls_preds = self.cls_head_camera(feature)
            reg_preds = self.reg_head_camera(feature)
            dir_preds = self.dir_head_camera(feature)
            print("Camera encoder.")
            
        elif 'processed_lidar' in data_dict:
            """
            LiDAR Encode
            """
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']


            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'batch_size': self.lidar_encoder.batch_size}

            batch_dict = self.lidar_encoder.vfe(batch_dict)
            batch_dict = self.lidar_encoder.spconv_block(batch_dict)
            batch_dict = self.lidar_encoder.map_to_bev(batch_dict) 
            lidar_feature_2d = self.lidar_encoder.ssfa(batch_dict['spatial_features'])
            lidar_feature_2d = self.shrink_lidar(lidar_feature_2d) 
            feature = lidar_feature_2d

            cls_preds = self.cls_head_lidar(feature)
            reg_preds = self.reg_head_lidar(feature)
            dir_preds = self.dir_head_lidar(feature)
            print("LiDAR encoder.")



        output_dict = {'cls_preds': cls_preds,
                       'reg_preds': reg_preds,
                       'dir_preds': dir_preds}


        return output_dict
