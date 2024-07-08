import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
import numpy as np
import torch

# New Import
from collections import OrderedDict
from opencood.utils.match import HungarianMatcher
from opencood.utils.residual_autoencoder import ResidualAutoEncoderV2ReconstructionLoss
from opencood.utils.bh_procedure import build_bh_procedure
from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
from opencood.utils import eval_utils
from .multiscale_attn_pgd import PGD
from .shift import Shift
from .rotate import Rotate
from .shift_and_rotate import Shift_and_Rotate
import os 
import time
import matplotlib.pyplot as plt

match_percentiles = {
    50: 0.03124090377241373,
    60: 0.0464456133544445,
    70: 0.07056263238191603,
    80: 0.12653420269489304,
    90: 0.21839267164468779,
    95: 0.3420280516147611
}

class CenterPointBaselineMultiscale(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(CenterPointBaselineMultiscale, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        self.voxel_size = args['voxel_size']
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range  = args['lidar_range']

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 8 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
        
        self.init_weight()

    def init_weight(self):
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -np.log((1 - pi) / pi) )
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    # Generate attack sources randomly
    def get_attack_src(self, agent_num, n_att):

        if agent_num - 1 < n_att:
            return []

        attacker = torch.randint(low=1,high=agent_num,size=(n_att,))

        tmp = []
        for i in range(len(attacker)):
            tmp.append(attacker[i])

        return tmp
    
    # Add attack in fuse module
    def att_fuse_module(self, feature_list, record_len, t_matrix, data_dict = None, attack = None, 
                attack_src = None, num = 0, attacked_feature = False, shift_feature = False, rotate_feature = False,
                attack_conf = None, real_data_dict = None, if_erase = False, erase_index = [], attack_target = 'pred', 
                pred_gt_box_tensor = None, dataset = None, shift_dir_of_box = [], if_fuse = True, if_inference = False):
        
        random_att = False
        attacked_feature_list = []
        residual_vector = None

        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):

            x = feature_list[i].clone()

            # save attacked feature (only level 1 now)
            if attacked_feature and i == 0:
                tmp_list = []
                for att in attack_src:
                    tmp_list.append(x[att])
                attacked_feature_list.append(tmp_list)

            # if shift or (erase + shift)
            if shift_feature and i == 0 and x.shape[0] > 1:
                tmp_list = []
                for att in attack_src:
                    tmp_list.append(x[att])
                attacked_feature_list.append(tmp_list)
                tmp_model = torch.nn.Conv2d(1,4,(2,3))
                shift_model = Shift(tmp_model, self.att_fuse_module, record_len, t_matrix, pairwise_t_matrix=self.pairwise_t_matrix, **attack_conf.attack.shift)

                tmp_att, _ = shift_model(real_data_dict, data_dict, if_attack_feat = True, attack_feat = attacked_feature_list, if_erase = if_erase, erase_index = erase_index, num = num, attack_conf = attack_conf, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box)
                x[att] = x[att] + tmp_att[0]
            
            # if rotate
            if rotate_feature and i == 0 and x.shape[0] > 1:
                tmp_list = []
                for att in attack_src:
                    tmp_list.append(x[att])
                attacked_feature_list.append(tmp_list)
                tmp_model = torch.nn.Conv2d(1,4,(2,3))
                rotate_model = Rotate(tmp_model, **attack_conf.attack.rotate)
                tmp_att, _ = rotate_model(real_data_dict, data_dict, if_attack_feat = True, attack_feat = attacked_feature_list)
                x[att] = x[att] + tmp_att[0]
            
            # If add random noise
            if random_att:
                if x.shape[0] > 1: 
                    # all agents are attackers
                    zero_part = torch.zeros(1, x.shape[1], x.shape[2], x.shape[3]).cuda()
                    perturb = torch.randn(x.shape[0] - 1, x.shape[1], x.shape[2], x.shape[3]).cuda()
                    perturb = torch.cat([zero_part, perturb], dim=0)
                    x = x + perturb
            else:
                # PGD attack
                # skip attack 
                if attack is None or len(attack_src) == 0:
                    nonsense = 12345
                # add attack to attacker 
                elif isinstance(attack, list):
                    num_att = len(attack_src)
                    # for each attacker
                    for j in range(num_att):
                        perturb = attack[i][j]
                        attacker = attack_src[j]
                        x[attacker] = x[attacker] + perturb
            

            if if_fuse:
                x_fuse = fuse_module(x, record_len, t_matrix)
            else:
                single_t_matrix = torch.tensor(
                    [[[[[1., 0., 0.],
                    [0., 1., 0.]],
                    [[1., 0., 0.],
                    [0., 1., 0.]]],
                    [[[1., 0., 0.],
                    [0., 1., 0.]],
                    [[1., 0., 0.],
                    [0., 1., 0.]]]]])
                x_fuse = fuse_module(x[0].unsqueeze(0), torch.tensor([1]), single_t_matrix)

            fused_feature_list.append(x_fuse)

            # Compute residual vector
            if i == 0:
                residual_vector = x_fuse - x[0]
        
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)
        
        if attacked_feature:
            return data_dict, attacked_feature_list

        return fused_feature, residual_vector
    
    # Add attack
    def forward(self, data_dict, attack=False, attack_target='pred', num = [0], delete_list = [], attack_type = 'pgd', dataset = None, if_single = False):

        '''
        attack: 攻击配置文件
        num: 列表包含一个batch中每个样本的idx
        delete_list,if_single: 不好删去的参数
        '''

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        t_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])
        self.pairwise_t_matrix = data_dict['pairwise_t_matrix']

        # spatial_features: (N, 64, 200, 504)
        spatial_features = batch_dict['spatial_features']

        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        # multiscale fusion without attack
        # step 1. make feature through encoder
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        batch_dict['feature_list'] = feature_list
        self.feature_list = feature_list

        # step 2. make feature fused by fuse modules
        single_t_matrix = torch.tensor(
            [[[[[1., 0., 0.],
            [0., 1., 0.]],
            [[1., 0., 0.],
            [0., 1., 0.]]],
            [[[1., 0., 0.],
            [0., 1., 0.]],
            [[1., 0., 0.],
            [0., 1., 0.]]]] for p in range(len(num))])
        fused_feature_list = []
        no_fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix))

            tmp_record_len = torch.tensor([1 for p in range(len(num))])
            # TODO: 这里不太确定feature_list的维度
            no_fused_feature_list.append(fuse_module(feature_list[i][:][0].unsqueeze(1), tmp_record_len, single_t_matrix))
        
        # step 3. make fused feature through decoder
        # spatial_features: (1, 384, 100, 252)
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 
        no_fused_feature = self.backbone.decode_multiscale_feature(no_fused_feature_list) 

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
            no_fused_feature = self.shrink_conv(no_fused_feature)

        cls = self.cls_head(fused_feature)
        bbox = self.reg_head(fused_feature)

        psm_single = self.cls_head(no_fused_feature)
        rm_single = self.reg_head(no_fused_feature)

        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)
        _, no_fuse_bbox_temp = self.generate_predicted_boxes(psm_single, rm_single)

        no_att_output_dict = {'cls_preds': cls,
                       'reg_preds': bbox_temp,
                       'bbox_preds': bbox}    # 计算loss的时候使用 'bbox', 在生成output的时候 'rm'
        
        without_attacker_output_dict = {'cls_preds': psm_single,
                'reg_preds': no_fuse_bbox_temp,
                'bbox_preds': rm_single}

        if self.use_dir:
            no_att_output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        # TODO: 不确定是否支持多batch
        # Generate pred outcome
        output_dict = OrderedDict()
        output_dict['ego'] = no_att_output_dict
        tmp_data_dict = OrderedDict()
        tmp_data_dict['ego'] = data_dict
        with torch.no_grad():
            pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
                dataset.post_process(tmp_data_dict, output_dict)

        ##########################################################
        evasion_list = []
        for s in range(len(num)):
            s_record_len = record_len[s].unsqueeze(0)
            s_t_matrix = t_matrix[s].unsqueeze(0)
            s_num = num[s]

            # TODO: 需要确认下batch_size > 1情况时这些变量的shape
            s_pred_gt_box_tensor = pred_gt_box_tensor[s]
            s_without_attacker_output_dict = self.new_dict(without_attacker_output_dict, s)
            s_no_att_output_dict = self.new_dict(no_att_output_dict, s)
            s_batch_dict = self.new_dict(batch_dict, s)
            s_data_dict = self.new_dict(data_dict, s)

            evasion = self.attack_process(attack, attack_type, s_record_len, s_t_matrix, s_batch_dict, s_data_dict, s_no_att_output_dict, dataset, s_without_attacker_output_dict, attack_target, s_num, s_pred_gt_box_tensor)

            evasion_list.append(evasion)
        
        return evasion_list
    
    def new_dict(self, your_dict, s):
        new_dict = {}
        for key, tensor in your_dict.items():
            new_dict[key] = tensor[s]
        return new_dict
        
    def attack_process(self, attack, attack_type, record_len, t_matrix, batch_dict, data_dict, no_att_output_dict, dataset, without_attacker_output_dict, attack_target, num, pred_gt_box_tensor):
        # Define attack model
        if isinstance(attack, bool) or attack == "TRUE":
            self.attack = attack or attack == "TRUE"   
        elif isinstance(attack, str):
            # yaml file 
            from omegaconf import OmegaConf
            attack_conf = OmegaConf.load(attack)

            if attack_type == 'pgd' or attack_type == 'shift_and_pgd' or attack_type == 'rotate_and_pgd' or attack_type == 'erase_and_shift_and_pgd':
                if attack_conf.attack is not None:
                    self.attack = True
                    self.attack_model = PGD(self.cls_head, self.att_fuse_module, self.reg_head, record_len, t_matrix, self.backbone, self.generate_predicted_boxes,**attack_conf.attack.pgd)
                    self.attack_target = attack_conf.attack.attack_target
                    n_att = attack_conf.attack.pgd.n_att
                    attack_srcs = self.get_attack_src(batch_dict['spatial_features'].shape[0], n_att)
                else:
                    self.attack = False
            elif attack_type == 'shift':
                if attack_conf.attack is not None:
                    self.attack = True
                    self.attack_model = Shift(self.fusion_net, self.att_fuse_module, record_len, t_matrix, data_dict['pairwise_t_matrix'], **attack_conf.attack.shift)
                    self.attack_target = attack_conf.attack.attack_target
                    n_att = attack_conf.attack.shift.n_att
                    attack_srcs = self.get_attack_src(batch_dict['spatial_features'].shape[0], n_att)
                else:
                    self.attack = False
            elif attack_type == 'erase_and_shift':
                if attack_conf.attack is not None:
                    self.attack = True
                    self.attack_model = Shift(self.fusion_net, self.att_fuse_module, record_len, t_matrix, data_dict['pairwise_t_matrix'], **attack_conf.attack.shift)
                    self.attack_target = attack_conf.attack.attack_target
                    n_att = attack_conf.attack.shift.n_att
                    attack_srcs = self.get_attack_src(batch_dict['spatial_features'].shape[0], n_att)
                else:
                    self.attack = False
            elif attack_type == 'rotate':
                if attack_conf.attack is not None:
                    self.attack = True
                    self.attack_model = Rotate(self.fusion_net, **attack_conf.attack.rotate)
                    self.attack_target = attack_conf.attack.attack_target
                    n_att = attack_conf.attack.rotate.n_att
                    attack_srcs = self.get_attack_src(batch_dict['spatial_features'].shape[0], n_att)
                else:
                    self.attack = False
            elif attack_type == 'shift_and_rotate':
                if attack_conf.attack is not None:
                    self.attack = True
                    self.attack_model = Shift_and_Rotate(self.fusion_net, attack_conf.attack.bbox_num, attack_conf.attack.shift.shift_length, attack_conf.attack.rotate.shift_angle, attack_conf.attack.n_att, attack_conf.attack.shift.shift_direction, attack_conf.attack.shift.padding_type, attack_conf.attack.rotate.shift_type, attack_conf.attack.rotate.padding_type)
                    self.attack_target = attack_conf.attack.attack_target
                    n_att = attack_conf.attack.n_att
                    attack_srcs = self.get_attack_src(batch_dict['spatial_features'].shape[0], n_att)
                else:
                    self.attack = False

        # (100, 252, 1, 7)
        anchors = data_dict['anchor_box']
        # (1, 100, 252, 7)
        reg_targets = data_dict['label_dict']['targets']
        # (1, 100, 252, 1)
        labels = data_dict['label_dict']['pos_equal_one']

        # If Attack
        if self.attack:

            # If pgd exists
            if attack_type == 'pgd' or attack_type == 'shift_and_pgd' or attack_type == 'rotate_and_pgd' or attack_type == 'erase_and_shift_and_pgd':

                if self.attack_model.attack_mode == 'self':
                    print("This occasion is not considered now!")
                    exit(0)
                else:
                    ref_result = no_att_output_dict

                if self.attack_target == 'gt':
                    att_reg_targets = reg_targets
                    att_labels = labels
                elif self.attack_target == 'pred':
                    att_reg_targets = ref_result['reg_preds'].reshape(1, 100, 252, 7).contiguous()
                    att_labels = ref_result['cls_preds'].permute(0, 2, 3, 1).contiguous()
                else:
                    raise NotImplementedError(self.attack_target)

            # If erasion exists, Generate Erase Index
            if attack_type == 'erase_and_shift_and_pgd' or attack_type == 'erase_and_shift':
                
                output_dict = OrderedDict()
                output_dict['ego'] = without_attacker_output_dict
                tmp_data_dict = OrderedDict()
                tmp_data_dict['ego'] = data_dict
                with torch.no_grad():
                    box_tensor, score, gt_box_tensor, bbox = \
                        dataset.post_process(tmp_data_dict, output_dict)

                if attack_target == 'gt':
                    erase_index = eval_utils.find_box_out(box_tensor, score, gt_box_tensor, attack_conf.attack.erase.iou_thresh)
                else:
                    erase_index = eval_utils.find_box_out(box_tensor, score, pred_gt_box_tensor, attack_conf.attack.erase.iou_thresh)
            
            # If shift, Generate shift directions
            if attack_type == 'erase_and_shift_and_pgd' and attack_conf.attack.shift.shift_direction == 'random':
                if attack_target == 'gt':
                    bbox_num = gt_box_tensor.shape[0] - len(erase_index)
                else:
                    bbox_num = pred_gt_box_tensor.shape[0] - len(erase_index)
                shift_dir_of_box = [np.random.randint(low=0,high=4) for k in range(bbox_num)]
            elif (attack_type == 'shift_and_pgd' or attack_type == 'shift' or attack_type == 'erase_and_shift') and attack_conf.attack.shift.shift_direction == 'random':
                if attack_target == 'gt':
                    bbox_num = gt_box_tensor.shape[0]
                else:
                    bbox_num = pred_gt_box_tensor.shape[0]
                shift_dir_of_box = [np.random.randint(low=0,high=4) for k in range(bbox_num)]

            if attack_type == 'pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, attack_srcs = attack_srcs)
            elif attack_type == 'shift':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box, num=num)
            elif attack_type == 'shift_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, pred_gt_box_tensor = pred_gt_box_tensor, shift_dir_of_box = shift_dir_of_box, dataset = dataset)
            elif attack_type == 'erase_and_shift_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, if_erase = True, erase_index = erase_index, dataset = dataset, pred_gt_box_tensor = pred_gt_box_tensor, shift_dir_of_box = shift_dir_of_box)
            elif attack_type == 'erase_and_shift':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs, if_erase = True, erase_index = erase_index, num = num, attack_conf = attack_conf, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box)
            elif attack_type == 'rotate_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = False, rotate_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, no_att_output_dict = no_att_output_dict)
            elif attack_type == 'rotate':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs)
            elif attack_type == 'shift_and_rotate':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs)

            return evasion, attack_src

    def generate_predicted_boxes(self, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        
        batch, H, W, code_size = box_preds.size()   ## code_size 表示的是预测的尺寸
        
        # batch_reg = box_preds[:, 0:2, :, :]  # x,y,z
        # batch_hei = box_preds[:, 2:3, :, :]
        # batch_dim = torch.exp(box_preds[:, 3:6, :, :])
        # # batch_dim = box_preds[:, 3:6, :, :]   # w h l 
        # batch_rots = box_preds[:, 6:7, :, :]
        # batch_rotc = box_preds[:, 7:8, :, :]
        # rot = torch.atan2(batch_rots, batch_rotc)
        
        
        box_preds = box_preds.reshape(batch, H*W, code_size)

        batch_reg = box_preds[..., 0:2]
        # batch_hei = box_preds[..., 2:3] 
        # batch_dim = torch.exp(box_preds[..., 3:6])
        
        h = box_preds[..., 3:4] * self.out_size_factor * self.voxel_size[0]
        w = box_preds[..., 4:5] * self.out_size_factor * self.voxel_size[1]
        l = box_preds[..., 5:6] * self.out_size_factor * self.voxel_size[2]
        batch_dim = torch.cat([h,w,l], dim=-1)
        batch_hei = box_preds[..., 2:3] * self.out_size_factor * self.voxel_size[2] + self.cav_lidar_range[2]

        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]

        rot = torch.atan2(batch_rots, batch_rotc)

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_lidar_range[0]   ## 基于feature_map 的size求解真实的坐标
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_lidar_range[1]


        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
        # batch_box_preds = batch_box_preds.reshape(batch, H, W, batch_box_preds.shape[-1])
        # batch_box_preds = batch_box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_box_preds_temp = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=1)
        # box_preds = box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_cls_preds = cls_preds.view(batch, H*W, -1)
        return cls_preds, batch_box_preds