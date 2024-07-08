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

from collections import OrderedDict
from opencood.utils.match import HungarianMatcher
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
from matplotlib import colors

from opencood.defense_model import RawAEDetector, ResidualAEDetector
from config.defense.percentiles import *


class CenterPointBaselineMultiscale(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args, temperature=20):
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
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i], temperature=temperature))
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

    # add evasion to feature_list
    def apply_attack(self, feature_list, evasion, attack_src, attack_type, record_len, t_matrix, attack_conf, real_data_dict, data_dict, erase_index, num, attack_target, pred_gt_box_tensor, dataset, shift_dir_of_box, gt_box_tensor):

        if 'shift' in attack_type:
            if_shift = True
        else:
            if_shift = False
        if 'erase' in attack_type:
            if_erase = True
        else:
            if_erase = False
        
        return_list = []

        for i in range(len(feature_list)):
            x = feature_list[i].clone()
            num_att = len(attack_src)

            # first shift
            if if_shift and x.shape[0] > 1:

                att_feat_list = [[],[],[]]
                for d in range(3):
                    if d == i:
                        for att in attack_src:
                            att_feat_list[d].append(x[att])
                    else:
                        att_feat_list[d].append(torch.zeros(C_list[d], H_list[d], W_list[d]).cuda())
                
                tmp_model = torch.nn.Conv2d(1,4,(2,3))
                shift_model = Shift(tmp_model, self.att_fuse_module, record_len, t_matrix, pairwise_t_matrix=self.pairwise_t_matrix, **attack_conf.attack.shift)

                tmp_x_att = x[att].clone().view(64, -1)
                value, count = torch.unique(tmp_x_att, dim=1, return_counts = True)
                max_index = torch.argmax(count)
                erase_value = value[:,max_index]

                if i in attack_conf.attack.shift.att_layer:
                    tmp_att_layer = [i]
                else:
                    tmp_att_layer = []
            
                tmp_att, _ = shift_model(real_data_dict, data_dict, if_attack_feat = True, attack_feat = att_feat_list, if_erase = if_erase, erase_index = erase_index, num = num, attack_conf = attack_conf, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box, erase_value = erase_value, gt_box_tensor = gt_box_tensor, att_layer = tmp_att_layer)
                x[att] = x[att] + torch.tensor(tmp_att[i][0]).cuda()

            # for each attacker
            for j in range(num_att):
                perturb = evasion[i][j]
                attacker = attack_src[j]
                x[attacker] = x[attacker] + perturb
            return_list.append(x)

        return return_list

    def generate_batch_dict(self, cav_content):
        
        voxel_features = cav_content['processed_lidar']['voxel_features']
        voxel_coords = cav_content['processed_lidar']['voxel_coords']
        voxel_num_points = cav_content['processed_lidar']['voxel_num_points']
        record_len = cav_content['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        _, _, H0, W0 = batch_dict['spatial_features'].shape
        t_matrix = normalize_pairwise_tfm(cav_content['pairwise_t_matrix'], H0, W0, self.voxel_size[0])
        
        spatial_features = batch_dict['spatial_features']
        if self.compression:
            spatial_features = self.naive_compressor
            (spatial_features)

        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        batch_dict['feature_list'] = feature_list
        
        return batch_dict, t_matrix, record_len
    
    def generate_erase_index(self, without_attacker_output_dict, data_dict, dataset, attack_target, attack_conf, pred_gt_box_tensor):
        
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

        return erase_index

    def visualize_tensor_both(self, tensor, attack_tensor, num, min_x, max_x, attack_conf, layer = 0):

        #  
        eps = f'{attack_conf.attack.pgd.eps}'
        attack = attack_conf.attack.save_path[12:]
        tmp_tensor1 = tensor.mean(dim=0)
        tmp_tensor1 = (tmp_tensor1 - min_x) / (max_x - min_x)
        tmp_tensor2 = attack_tensor.mean(dim=0)
        tmp_tensor2 = (tmp_tensor2 - min_x) / (max_x - min_x)

        v_min = min(torch.min(tmp_tensor1), torch.min(tmp_tensor2))
        v_max = max(torch.max(tmp_tensor1), torch.max(tmp_tensor2))
        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        # norm = colors.Normalize(vmin=0, vmax=5)

        import seaborn as sns

        fig, (ax1, ax2) = plt.subplots(2, 1)

        heatmap1 = ax1.imshow(tmp_tensor1.detach().cpu().numpy(), cmap='viridis', norm=norm)
        ax1.axis('off')

        heatmap2 = ax2.imshow(tmp_tensor2.detach().cpu().numpy(), cmap='viridis', norm=norm)
        ax2.axis('off')

        position = fig.add_axes([0.92, 0.12, 0.015, .78 ]) #位置[左,下,右,上]
        cbar = fig.colorbar(heatmap1, ax=[ax1, ax2], cax=position)

        save_path = 'outcome/feature_visualize/both/'
        save_path = save_path + f'eps{eps}_{attack}_layer{layer}_step_80_PGD/'
        fig.subplots_adjust(hspace=0.2)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(save_path + f'sample_{num}.png', dpi = 300)
        plt.close()

    def visualize_tensor_single(self, tensor, attack_tensor, num, min_x, max_x, attack_conf, layer = 0):

        #  
        eps = f'{attack_conf.attack.pgd.eps}'
        attack = 'pgd'
        tmp_tensor1 = tensor.mean(dim=0)
        tmp_tensor1 = (tmp_tensor1 - min_x) / (max_x - min_x)
        tmp_tensor2 = attack_tensor.mean(dim=0)
        tmp_tensor2 = (tmp_tensor2 - min_x) / (max_x - min_x)

        v_min = min(torch.min(tmp_tensor1), torch.min(tmp_tensor2))
        v_max = max(torch.max(tmp_tensor1), torch.max(tmp_tensor2))
        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        # norm = colors.Normalize(vmin=0, vmax=5)

        import seaborn as sns

        save_path = 'outcome/feature_visualize/single—pgd/'
        save_path = save_path + f'eps{eps}_{attack}_layer{layer}_step_80/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig = plt.figure(figsize=(63,25))
        axes = fig.add_axes([0,0,1,1])
        axes.set_axis_off()
        axes.imshow(tmp_tensor1.detach().cpu().numpy(), cmap='viridis', aspect='auto')
        plt.savefig(save_path + f'without_att_sample_{num}.png', bbox_inches='tight', dpi=300)
        plt.close()

        fig = plt.figure(figsize=(63,25))
        axes = fig.add_axes([0,0,1,1])
        axes.set_axis_off()
        axes.imshow(tmp_tensor2.detach().cpu().numpy(), cmap='viridis', aspect='auto')
        plt.savefig(save_path + f'with_att_sample_{num}.png', bbox_inches='tight', dpi=300)
        plt.close()

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

        # attacker = torch.randint(low=0,high=1,size=(n_att,))
        tmp = []
        for i in range(len(attacker)):
            tmp.append(attacker[i])

        return tmp
    
    # Add attack in fuse module
    def att_fuse_module(self, feature_list, record_len, t_matrix, data_dict = None, attack = None, 
                attack_src = None, num = 0, attacked_feature = False, shift_feature = False, rotate_feature = False,
                attack_conf = None, real_data_dict = None, if_erase = False, erase_index = [], attack_target = 'pred', 
                pred_gt_box_tensor = None, dataset = None, shift_dir_of_box = [], if_fuse = True, if_inference = False, 
                gt_box_tensor = None, cls = None, if_att_score = False, if_shift_attack = False, shift_attack = None):
        
        random_att = False
        residual_vector = None

        fused_feature_list = []
        attacked_feature_list = []
        compare_fused_feature_list = []
        if shift_attack == None:
            shift_attack = []
        for i, fuse_module in enumerate(self.fusion_net):
            
            
            x = feature_list[i].clone()
            # if x.shape[0] > 1:
                # attacker_normal = x[1].clone()
                # min_x, max_x = x[1].mean(dim=0).min(), x[1].mean(dim=0).max()

            # save attacked feature (only level 1 now)
            if attacked_feature:
                tmp_list = []
                for att in attack_src:
                    tmp_list.append(x[att])
                attacked_feature_list.append(tmp_list)

            # if shift or (erase + shift)
            if shift_feature and x.shape[0] > 1:

                if shift_attack == None or len(shift_attack) != 3:
                    tmp_list = []
                    att_feat_list = [[],[],[]]
                    for d in range(3):
                        if d == i:
                            for att in attack_src:
                                att_feat_list[d].append(x[att])
                        else:
                            att_feat_list[d].append(torch.zeros(C_list[d], H_list[d], W_list[d]).cuda())
                    tmp_model = torch.nn.Conv2d(1,4,(2,3))
                    shift_model = Shift(tmp_model, self.att_fuse_module, record_len, t_matrix, pairwise_t_matrix=self.pairwise_t_matrix, **attack_conf.attack.shift)

                    tmp_x_att = x[att].clone().view(64, -1)
                    value, count = torch.unique(tmp_x_att, dim=1, return_counts = True)
                    max_index = torch.argmax(count)
                    erase_value = value[:,max_index]
                    
                    if i in attack_conf.attack.shift.att_layer:
                        tmp_att_layer = [i]
                    else:
                        tmp_att_layer = []

                    tmp_att, _ = shift_model(real_data_dict, data_dict, if_attack_feat = True, attack_feat = att_feat_list, if_erase = if_erase, erase_index = erase_index, num = num, attack_conf = attack_conf, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box, erase_value = erase_value, gt_box_tensor = gt_box_tensor, att_layer = tmp_att_layer)
                    min_x, max_x = x[att].mean(dim=0).min(), x[att].mean(dim=0).max()
                    shift_x_clone = x[att].clone()
                    x[att] = x[att] + torch.tensor(tmp_att[i][0]).cuda()
                    shift_attack.append(torch.tensor(tmp_att[i][0]).cuda())
                    # if i == 0 and if_inference:
                    #     self.visualize_tensor_both(shift_x_clone, x[att], num, min_x, max_x)
                else:
                    for att in attack_src:
                        x[att] = x[att] + shift_attack[i]
            
            if if_shift_attack and i == 2:
                return shift_attack
                    
            
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

                        attacker_normal = x[1].clone()
                        min_x, max_x = x[1].mean(dim=0).min(), x[1].mean(dim=0).max()
                        
                        if isinstance(perturb, np.ndarray):
                            perturb = torch.tensor(perturb).cuda()
                        x[attacker] = x[attacker] + perturb
                        min_x, max_x = min(min_x, x[attacker].mean(dim=0).min()), max(max_x, x[attacker].mean(dim=0).max())

                        if if_inference:
                            # self.visualize_tensor_both(attacker_normal, x[attacker], num, min_x, max_x, attack_conf, i)
                            self.visualize_tensor_single(attacker_normal, x[attacker], num, min_x, max_x, attack_conf, i)

            x_clone = x.clone()
            if if_fuse:
                if (not attacked_feature) and not if_inference:
                    x_fuse, attention_score = fuse_module(x, record_len, t_matrix, num = num, 
                        if_draw = if_inference, cls = cls, if_att_score = if_att_score)
                elif if_inference:
                    x_fuse = fuse_module(x, record_len, t_matrix, num = num, 
                        if_draw = if_inference, cls = cls)
                else:
                    x_fuse = fuse_module(x, record_len, t_matrix, num = num)
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
        
        if if_shift_attack:
            return fused_feature, shift_attack

        if if_att_score:
            if if_inference:
                attention_score = None      
            return fused_feature, residual_vector, attention_score
        else:
            return fused_feature, residual_vector,
    
    # Add attack
    def forward(self, data_dict, attack=None, attack_target='pred', save_path=None, num = 0, save_attack = False, delete_list = [], attack_type = 'pgd', dataset = None, if_single = False, save_dir = None, if_att_time = False):

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
            [0., 1., 0.]]]]])
        fused_feature_list = []
        no_fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
            if record_len[0] == 2:
                tmp_record_len = torch.tensor([1])
                no_fused_feature_list.append(fuse_module(feature_list[i][0].unsqueeze(0), tmp_record_len, single_t_matrix, if_draw = 'no_fuse'))
            else:
                no_fused_feature_list = fused_feature_list.copy()
        
        # step 3. make fused feature through decoder
        # spatial_features: (1, 384, 100, 252)
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 
        no_fused_feature = self.backbone.decode_multiscale_feature(no_fused_feature_list) 

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
            no_fused_feature = self.shrink_conv(no_fused_feature)

        cls = self.cls_head(fused_feature)
        bbox = self.reg_head(fused_feature)

        # 如果无攻击情况下可视化attention score
        if attack == None:
            for i in range(3):
                fuse_module(feature_list[i], record_len, t_matrix, num = num, if_draw = True, cls = cls)
        else:
            suzuki = 1

        psm_single = self.cls_head(no_fused_feature)
        rm_single = self.reg_head(no_fused_feature)

        # 把bbox 的第二维度变成7 
        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)
        _, no_fuse_bbox_temp = self.generate_predicted_boxes(psm_single, rm_single)

        # print(bbox.equal(bbox_temp))
        no_att_output_dict = {'cls_preds': cls,
                       'reg_preds': bbox_temp,
                       'bbox_preds': bbox}    # 计算loss的时候使用 'bbox', 在生成output的时候 'rm'
        
        without_attacker_output_dict = {'cls_preds': psm_single,
                'reg_preds': no_fuse_bbox_temp,
                'bbox_preds': rm_single}

        if self.use_dir:
            no_att_output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        #####################################################

        # Generate pred outcome
        output_dict = OrderedDict()
        output_dict['ego'] = no_att_output_dict
        tmp_data_dict = OrderedDict()
        tmp_data_dict['ego'] = data_dict
        with torch.no_grad():
            pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
                dataset.post_process(tmp_data_dict, output_dict)
        
        if pred_gt_box_tensor == None:
            return no_att_output_dict, None
        
        time1 = time.time()
        # Define attack model  
        if attack == None:
            self.attack = False
        elif isinstance(attack, str):
            # yaml file 
            from omegaconf import OmegaConf
            attack_conf = OmegaConf.load(attack)

            if attack_type == 'pgd' or attack_type == 'shift_and_pgd' or attack_type == 'erase_and_shift_and_pgd':
                if attack_conf.attack is not None:
                    self.attack = True
                    self.attack_model = PGD(self.cls_head, self.att_fuse_module, self.reg_head, record_len, t_matrix, self.backbone, self.generate_predicted_boxes,**attack_conf.attack.pgd)
                    self.attack_target = attack_conf.attack.attack_target
                    n_att = attack_conf.attack.n_att
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

        # (100, 252, 1, 7)
        anchors = data_dict['anchor_box']
        # (1, 100, 252, 7)
        reg_targets = data_dict['label_dict']['targets']
        # (1, 100, 252, 1)
        labels = data_dict['label_dict']['pos_equal_one']

        
        # If Attack
        if self.attack:

            # If pgd exists
            if attack_type == 'pgd' or attack_type == 'shift_and_pgd' or attack_type == 'erase_and_shift_and_pgd':

                if self.attack_model.attack_mode == 'self':
                    print("This occasion is not considered now!")
                    exit(0)
                else:
                    ref_result = no_att_output_dict

                if self.attack_target == 'gt':
                    att_reg_targets = reg_targets # (1, 100, 252, 7)
                    att_labels = labels # (1, 100, 252, 1)
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
            shift_dir_of_box = []
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

            # If shift, Generate shift directions
            if 'shift' in attack_type:

                shift_dir_save_path = attack_conf.attack.shift.shift_dir_save_path
                
                shift_dir_save_path = shift_dir_save_path + f'/{attack_conf.attack.pgd.eps}/'
                if os.path.exists(shift_dir_save_path) == False:
                    os.makedirs(shift_dir_save_path)
                np.save(shift_dir_save_path + f'sample_{num}.npy', shift_dir_of_box)

            if attack_type == 'pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, dataset = dataset, pred_gt_box_tensor = pred_gt_box_tensor, gt_box_tensor = gt_box_tensor)
            elif attack_type == 'shift':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box, num=num, gt_box_tensor = gt_box_tensor, att_layer = attack_conf.attack.shift.att_layer)
            elif attack_type == 'shift_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, pred_gt_box_tensor = pred_gt_box_tensor, shift_dir_of_box = shift_dir_of_box, dataset = dataset, gt_box_tensor = gt_box_tensor, att_layer = attack_conf.attack.shift.att_layer)
            elif attack_type == 'erase_and_shift_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, if_erase = True, erase_index = erase_index, dataset = dataset, pred_gt_box_tensor = pred_gt_box_tensor, shift_dir_of_box = shift_dir_of_box, gt_box_tensor = gt_box_tensor, att_layer = attack_conf.attack.shift.att_layer)
            elif attack_type == 'erase_and_shift':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs, if_erase = True, erase_index = erase_index, num = num, attack_conf = attack_conf, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box, gt_box_tensor = gt_box_tensor, att_layer = attack_conf.attack.shift.att_layer)

            # Save attack 
            if save_attack:

                tmp = {'block1':torch.clone(evasion[0]).detach().cpu().numpy(),'block2':torch.clone(evasion[1]).detach().cpu().numpy(),'block3':torch.clone(evasion[2]).detach().cpu().numpy(),'src':attack_src}

                if attack_type == 'pgd':
                    if not os.path.exists(save_path + f"_{attack_conf.attack.pgd.eps[0]}"):
                        os.makedirs(save_path + f"_{attack_conf.attack.pgd.eps[0]}")
                    np.save(save_path + f"_{attack_conf.attack.pgd.eps[0]}" + f'/sample_{num}.npy', tmp)
                elif attack_type == 'shift':
                    if not os.path.exists(save_path + f"_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length"):
                        os.makedirs(save_path + f"_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length")
                    np.save(save_path + f"_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length" + f'/sample_{num}.npy', tmp)
                elif attack_type == 'shift_and_pgd':
                    tmp_save_path = save_path + f"_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}"
                    if not os.path.exists(tmp_save_path):
                        os.makedirs(tmp_save_path)
                    np.save(tmp_save_path + f'/sample_{num}.npy', tmp)
                elif attack_type == 'erase_and_shift_and_pgd':
                    tmp_save_path = save_path + f"_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}_iou{attack_conf.attack.erase.iou_thresh}"
                    if not os.path.exists(tmp_save_path):
                        os.makedirs(tmp_save_path)
                    np.save(tmp_save_path + f'/sample_{num}.npy', tmp)
                elif attack_type == 'erase_and_shift':
                    tmp_save_path = save_path + f"_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_iou{attack_conf.attack.erase.iou_thresh}"
                    if not os.path.exists(tmp_save_path):
                        os.makedirs(tmp_save_path)
                    np.save(tmp_save_path + f'/sample_{num}.npy', tmp)

            # import ipdb; ipdb.set_trace
            if attack_type == 'shift_and_pgd':
                fused_feature_1, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor)
            elif attack_type == 'erase_and_shift_and_pgd':
                fused_feature_1, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, if_erase = True, erase_index = erase_index, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor, cls = cls)
            else:
                fused_feature_1, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num, delete_list=delete_list, if_inference = True, cls = cls, attack_conf = attack_conf, real_data_dict = data_dict, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, gt_box_tensor = gt_box_tensor)

            if self.shrink_flag:
                fused_feature_1 = self.shrink_conv(fused_feature_1)
            
            cls_1 = self.cls_head(fused_feature_1)
            bbox_1 = self.reg_head(fused_feature_1)

            _, bbox_temp_1 = self.generate_predicted_boxes(cls_1, bbox_1)


            att_output_dict = {'cls_preds': cls_1,
                        'reg_preds': bbox_temp_1,
                        'bbox_preds': bbox_1
                        }

            _, bbox_temp_single = self.generate_predicted_boxes(psm_single, rm_single)

            att_output_dict.update({'cls_preds_single': psm_single,
                        'reg_preds_single': bbox_temp_single,
                        'bbox_preds_single': rm_single,
                        })
        else:
            att_output_dict = {'cls_preds': None, 'reg_preds': None,
            'bbox_preds': None}
            
        spent_time = time.time() - time1
        
        if if_att_time:
            if self.attack and (attack_type == 'erase_and_shift' or attack_type == 'erase_and_shift_and_pgd'):
                return att_output_dict, erase_index, spent_time
            elif self.attack:
                return att_output_dict, None, spent_time
            else:
                if if_single:
                    no_att_output_dict = without_attacker_output_dict
                return no_att_output_dict, None, spent_time

        if self.attack and (attack_type == 'erase_and_shift' or attack_type == 'erase_and_shift_and_pgd'):
            return att_output_dict, erase_index
        elif self.attack:
            return att_output_dict, None
        else:
            if if_single:
                no_att_output_dict = without_attacker_output_dict
            return no_att_output_dict, None

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