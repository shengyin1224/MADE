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

match_percentiles = {
    95: 0.5737447053194045
}

residual_ae_percentiles = {
    95: 92569.8688802083
}

raw_ae_percentiles = {
    0: 822089.00625,
    1: 169663.17369791667,
    2: 21659.621158854156
}

C_list = [64, 128, 256]
H_list = [100, 50, 25]
W_list = [252, 126, 63]

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

    def attack_detection(self, data_dict, dataset, method, attack = None, num = 0, save_dir = 'test_set_pred', attack_type = 'pgd', attack_conf = None, defense_model = None):
        if method == 'match':
            return self.attack_detection_match(data_dict, dataset, attack, num = num, save_dir=save_dir, attack_type = attack_type, attack_conf = attack_conf)
        elif method == 'residual' or method == 'raw':
            return self.attack_detection_ae(data_dict, dataset, attack, num = num, save_dir = save_dir, attack_type = attack_type,  method=method, attack_conf=attack_conf, defense_model = defense_model)
        elif method == 'multi_test':
            return self.attack_detection_multi_test(data_dict, dataset, attack, num = num, attack_type=attack_type, attack_conf=attack_conf, save_dir = save_dir)
        else:
            print("Please choose an available method among match,autoencoder and multi_test!")
            exit(0)
    

    def generate_ae_train(self, data_dict, save_dir = 'validation', num = 0, ae_type = 'residual', attack = None, attack_type = 'pgd', attack_conf = None, dataset = None):
        """
        针对训练集生成feature maps
        """
        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        ae_path = 'generate_ae_loss/' + ae_type + '/' + save_dir

        # ego-data process
        ego_output, _ = self.forward(data_dict=cav_content, if_single=True, dataset=dataset)

        # 生成 batch_dict
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

        # fuse-data process
        if num_agent == 1:
            # 不存储任何数据防止干扰正常数据
            return

        # 生成攻击样本
        if attack != None:
            if attack_type != 'no attack':
                attack = attack.item()
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]

            attack_target = attack_conf.attack.attack_target

            # 生成pred_gt_box_tensor
            no_att_output_dict, _ = self.forward(data_dict=cav_content, dataset=dataset)
            output_dict = OrderedDict()
            output_dict['ego'] = no_att_output_dict
            with torch.no_grad():
                pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
                dataset.post_process(data_dict, output_dict)
            
            if attack_type == 'erase_and_shift_and_pgd':
                erase_index = self.generate_erase_index(ego_output, data_dict['ego'], dataset, attack_target, attack_conf, pred_gt_box_tensor)
            else:
                erase_index = None

            if 'shift' in attack_type and attack_conf.attack.shift.shift_direction == 'random':
                # folder_path_shift = 'outcome/shift_dir_of_box/1013/[1.5, 1.5, 0]'
                # shift_dir_of_box = np.load(folder_path_shift + f'sample_{num}.npy')
                if attack_type == 'erase_and_shift_and_pgd':
                    bbox_num = pred_gt_box_tensor.shape[0] - len(erase_index)
                else:
                    bbox_num = pred_gt_box_tensor.shape[0]
                shift_dir_of_box = [np.random.randint(low=0,high=4) for k in range(bbox_num)]
            else:
                shift_dir_of_box = None

            attacked_feature_list = self.apply_attack(feature_list, actual_attack, attack_src, attack_type, record_len, t_matrix, attack_conf, data_dict['ego'], batch_dict, erase_index, num, attack_target, pred_gt_box_tensor, dataset, shift_dir_of_box, gt_box_tensor)

            feature_list = attacked_feature_list

        if ae_type == 'raw':

            for k in range(3):
                if not os.path.exists(ae_path + f'/layer_{k}'):
                    os.makedirs(ae_path + f'/layer_{k}')

                save_feature = feature_list[k][1].detach().cpu().numpy()
                # print(save_feature.sum())
                np.save(ae_path + f'/layer_{k}/sample_{num}.npy',save_feature)
                
        else:

            add_zero_feature_list = [j.clone() for j in feature_list]
            for i in range(3):
                add_zero_feature_list[i][1] = torch.zeros_like(add_zero_feature_list[i][1], device='cuda')
                
            fused_feature_list = []
            no_fused_feature_list = []
            for i, fuse_module in enumerate(self.fusion_net):
                fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
                no_fused_feature_list.append(fuse_module(add_zero_feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))

                # import ipdb; ipdb.set_trace()
            
            fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list).squeeze(0) 
            no_fused_feature = self.backbone.decode_multiscale_feature(no_fused_feature_list).squeeze(0)

            save_feature = (fused_feature - no_fused_feature).detach().cpu().numpy()

            if os.path.exists(ae_path) == False:
                os.makedirs(ae_path)

            np.savez(ae_path + f'/sample_{num}.npz', save_feature = save_feature)

        return    

    def generate_ae_val(self, data_dict, save_dir = 'test_set_pred', num = 0, ae_type = 'residual', defense_model = None):
        """
        针对Validation Set生成Score
        """

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        if ae_type == 'residual':
            percentiles = residual_ae_percentiles
        else:
            percentiles = raw_ae_percentiles
        ae_path = 'generate_ae_loss/' + ae_type + '/' + save_dir

        if num_agent == 1:
            return

        # 生成 batch_dict
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

        if not os.path.exists(ae_path):
            os.makedirs(ae_path)
        
        # fuse-data process
        attacked_feature_list = feature_list

        if ae_type == 'raw':

            self.defense_model_0 = defense_model[0]
            self.defense_model_1 = defense_model[1]
            self.defense_model_2 = defense_model[2]
            
            mean_feature_0 = torch.from_numpy(np.load(f'generate_ae_loss/raw/trainset_1017_from_test/Layer0_raw_mean_feature.npy')).cuda()
            mean_feature_1 = torch.from_numpy(np.load(f'generate_ae_loss/raw/trainset_1017_from_test/Layer1_raw_mean_feature.npy')).cuda()
            

            id_keep, additional_dict_0 = self.defense_model_0(
                                        attacked_feature_list[0] - mean_feature_0.unsqueeze(0), 
                                        record_len,
                                        [torch.arange(1,2).cuda()],
                                        [torch.ones(1).cuda()],layer_num = 0)
            id_keep, additional_dict_1 = self.defense_model_1(
                                        attacked_feature_list[1] - mean_feature_1.unsqueeze(0), 
                                        record_len,
                                        [torch.arange(1,2).cuda()],
                                        [torch.ones(0).cuda()],layer_num = 1)
            id_keep, additional_dict_2 = self.defense_model_2(
                                        attacked_feature_list[2], 
                                        record_len,
                                        [torch.arange(1,2).cuda()],
                                        [torch.ones(0).cuda()],layer_num = 2)

            score_list = [additional_dict_0['score'], additional_dict_1['score'], additional_dict_2['score']]
            

            np.save(ae_path + f'/sample_{num}.npy', score_list)
        
        else:

            add_zero_feature_list = [j.clone() for j in feature_list]
            for i in range(3):
                add_zero_feature_list[i][1] = torch.zeros_like(add_zero_feature_list[i][1], device='cuda')

            fused_feature_list = []
            no_fused_feature_list = []
            for i, fuse_module in enumerate(self.fusion_net):
                fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
                no_fused_feature_list.append(fuse_module(add_zero_feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None)) 

                    
            fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list).squeeze(0) 
            no_fused_feature = self.backbone.decode_multiscale_feature(no_fused_feature_list).squeeze(0)

            residual_feature = (fused_feature - no_fused_feature)

            # centralize
            mean_feature = torch.from_numpy(np.load('generate_ae_loss/residual/1017_res_mean_feature.npy')).cuda()
            residual_feature = residual_feature - mean_feature.unsqueeze(0)

            self.defense_model = defense_model
            # self.defense_model = ResidualAEDetector(384, ckpt="opencood/ae_train/1016-RES-AE-train/autoencoder_res_199.pth", threshold = residual_ae_percentiles[95]).cuda()

            id_keep, additional_dict = self.defense_model(
                                        residual_feature,
                                        record_len,
                                        t_matrix,
                                        [torch.arange(1,2).cuda()],
                                        [torch.ones(0).cuda()])
                                        
            
            score = [additional_dict['score']]

            np.save(ae_path + f'/sample_{num}.npy', score)

        return

    def attack_detection_ae(self, data_dict, dataset, attack = None, method = 'residual', save_dir = 'test_set_pred', num = 0, attack_type = 'pgd', attack_conf = None, defense_model = None):

        ae_type = method
        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        if ae_type == 'residual':
            percentiles = residual_ae_percentiles
        else:
            percentiles = raw_ae_percentiles
        ae_path = 'generate_ae_loss/' + ae_type + '/'

        # ego-data process
        ego_output, _ = self.forward(data_dict=cav_content, if_single=True, dataset=dataset)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)

        # 生成 batch_dict
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

        if not os.path.exists(ae_path + save_dir):
            os.makedirs(ae_path + save_dir)
        
        # fuse-data process
        if num_agent == 1:
            np.save(ae_path + save_dir + f'/sample_{num}.npy',[])
            return target_box_tensor, target_score, gt_box_tensor, []

        if attack_type != 'no attack':
            attack = attack.item()
        attack_src = attack['src']
        if attack_src == []:
            actual_attack = None
        else:
            actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
        
        attack_target = attack_conf.attack.attack_target

        # 生成pred_gt_box_tensor
        no_att_output_dict, _ = self.forward(data_dict=cav_content, dataset=dataset)
        output_dict = OrderedDict()
        output_dict['ego'] = no_att_output_dict
        with torch.no_grad():
            pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
            dataset.post_process(data_dict, output_dict)
        
        if attack_type == 'erase_and_shift_and_pgd':
            erase_index = self.generate_erase_index(ego_output, data_dict['ego'], dataset, attack_target, attack_conf, pred_gt_box_tensor)
        else:
            erase_index = None

        if 'shift' in attack_type and attack_conf.attack.shift.shift_direction == 'random':
            # folder_path_shift = 'outcome/shift_dir_of_box/1013/[1.5, 1.5, 0]'
            folder_path_shift = 'outcome/shift_dir_of_box/1202_train/[1.5, 1.5, 0]'
            shift_dir_of_box = np.load(folder_path_shift + f'sample_{num}.npy')
        else:
            shift_dir_of_box = None

        attacked_feature_list = self.apply_attack(feature_list, actual_attack, attack_src, attack_type, record_len, t_matrix, attack_conf, data_dict['ego'], batch_dict, erase_index, num, attack_target, pred_gt_box_tensor, dataset, shift_dir_of_box, gt_box_tensor)

        if attack_type == 'no attack': 
            gt_attacker_label = [torch.ones(0).cuda()]
        else:
            gt_attacker_label = [torch.ones(1).cuda()]
        
        if ae_type == 'residual':
            self.defense_model_0 = defense_model

            add_zero_feature_list = [j.clone() for j in attacked_feature_list]
            for i in range(3):
                add_zero_feature_list[i][1] = torch.zeros_like(add_zero_feature_list[i][1], device='cuda')
            fused_feature_list = []
            no_fused_feature_list = []
            for i, fuse_module in enumerate(self.fusion_net):
                fused_feature_list.append(fuse_module(attacked_feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
                no_fused_feature_list.append(fuse_module(add_zero_feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
            
            fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list).squeeze(0) 
            no_fused_feature = self.backbone.decode_multiscale_feature(no_fused_feature_list).squeeze(0)

            residual_feature = (fused_feature - no_fused_feature).unsqueeze(0)
            residual_feat_layer0 = (fused_feature_list[0] - no_fused_feature_list[0])
            residual_feat_layer1 = (fused_feature_list[1] - no_fused_feature_list[1])
            
            # residual_feature = (fused_feature_list[0] - no_fused_feature_list[0
            # ])

            # centralize
            # mean_feature = torch.from_numpy(np.load('opencood/ae_train/1010-AE-train/layer0/layer_0.npy')).cuda()
            # mean_feature = torch.from_numpy(np.load('generate_ae_loss/residual/1016_res_mean_feature.npy')).cuda()
            # residual_feature = residual_feature - mean_feature.unsqueeze(0)

            # vis_feature
            # import pdb; pdb.set_trace()
            overall_path = 'outcome/vis_residual_1216/' + 'no_attack_temp20_'
            residual_vis_path = overall_path + 'after_decoder'
            if not os.path.exists(residual_vis_path):
                os.makedirs(residual_vis_path)
            residual_vis_feature = residual_feature.clone().detach().cpu().numpy()
            plt.imshow(residual_vis_feature.squeeze(0).mean(axis=0))
            plt.colorbar()
            plt.savefig(residual_vis_path + f'/sample_{num}.png')
            plt.close()

            residual_vis_path = overall_path + 'layer0'
            if not os.path.exists(residual_vis_path):
                os.makedirs(residual_vis_path)
            residual_vis_feature = residual_feat_layer0.clone().detach().cpu().numpy()
            plt.imshow(residual_vis_feature.squeeze(0).mean(axis=0))
            plt.colorbar()
            plt.savefig(residual_vis_path + f'/sample_{num}.png')
            plt.close()

            residual_vis_path = overall_path + 'layer1'
            if not os.path.exists(residual_vis_path):
                os.makedirs(residual_vis_path)
            residual_vis_feature = residual_feat_layer1.clone().detach().cpu().numpy()
            plt.imshow(residual_vis_feature.squeeze(0).mean(axis=0))
            plt.colorbar()
            plt.savefig(residual_vis_path + f'/sample_{num}.png')
            plt.close()

            id_keep, additional_dict = self.defense_model_0(
                # feature_list,
                residual_feature, 
                record_len,
                t_matrix,
                [torch.arange(1,2).cuda()],
                gt_attacker_label)
        else:
        
            self.defense_model_0 = defense_model[0]
            self.defense_model_1 = defense_model[1]
            self.defense_model_2 = defense_model[2]    
            
            mean_feature_0 = torch.from_numpy(np.load(f'generate_ae_loss/raw/trainset_1016_new/Layer0_raw_mean_feature.npy')).cuda()
            mean_feature_1 = torch.from_numpy(np.load(f'generate_ae_loss/raw/trainset_1016_new/Layer1_raw_mean_feature.npy')).cuda()
            

            id_keep_0, additional_dict_0 = self.defense_model_0(
                                        attacked_feature_list[0] - mean_feature_0.unsqueeze(0), 
                                        record_len,
                                        [torch.arange(1,2).cuda()],
                                        [torch.ones(1).cuda()],layer_num = 0)
            id_keep_1, additional_dict_1 = self.defense_model_1(
                                        attacked_feature_list[1] - mean_feature_1.unsqueeze(0), 
                                        record_len,
                                        [torch.arange(1,2).cuda()],
                                        [torch.ones(0).cuda()],layer_num = 1)
            id_keep_2, additional_dict_2 = self.defense_model_2(
                                        attacked_feature_list[2], 
                                        record_len,
                                        [torch.arange(1,2).cuda()],
                                        [torch.ones(0).cuda()],layer_num = 2)
        
            if additional_dict_0['correct'] == 0:
                id_keep = id_keep_0
            elif additional_dict_1['correct'] == 0:
                id_keep = id_keep_1
            # elif additional_dict_2['correct'] == 0:
            #     id_keep = id_keep_2
            else:
                id_keep = id_keep_0
        
        if 1 in id_keep[0]:
            pred = 0
        else:
            pred = 1

        if ae_type == 'raw': 
            save_score = [additional_dict_0['score'], additional_dict_1['score'], additional_dict_2['score']]
            # save_score = additional_dict_0['score']
        else:
            save_score = additional_dict['score']
        
        if attack_type == 'no attack':
            # np.save(ae_path + save_dir + f'/sample_{num}.npy',np.array([[save_score, 0]], dtype = object))
            label = 0
        else:
            # np.save(ae_path + save_dir + f'/sample_{num}.npy',np.array([[save_score, 1]], dtype = object))
            label = 1
        
        new_feature_list, new_record_len, new_t_matrix = self.defense_model_0.del_attacker(id_keep, attacked_feature_list, record_len, t_matrix)

        fused_att_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_att_feature_list.append(fuse_module(new_feature_list[i], new_record_len, torch.stack(new_t_matrix)))
        
        fused_att_feature = self.backbone.decode_multiscale_feature(fused_att_feature_list)

        fused_feature_1 = fused_att_feature
        if self.shrink_flag:
            fused_feature_1 = self.shrink_conv(fused_feature_1)
            
        cls_1 = self.cls_head(fused_feature_1)
        bbox_1 = self.reg_head(fused_feature_1)
        _, bbox_temp_1 = self.generate_predicted_boxes(cls_1, bbox_1)

        att_output_dict = {'cls_preds': cls_1,
                    'reg_preds': bbox_temp_1,
                    'bbox_preds': bbox_1
                    }
        output_dict = OrderedDict()
        output_dict['ego'] = att_output_dict

        with torch.no_grad():
            box_tensor, score, gt_box_tensor, bbox = \
                dataset.post_process(data_dict, output_dict)
        
        return box_tensor, score, gt_box_tensor, [pred, label]


    def generate_match_loss(self, data_dict, dataset, attack = None, num = 0, save_dir = 'valiadation'):
        """
        inputs:
            data_dict: origin data which includes other agents' information
            attack: (Dict) have 4 parts -- attack_src, block 1, block 2, block 3
            method: detection method like "match_cost","raw autoencoder" and so on
            dataset: used for the data post_process
        """

        match_path = 'generate_match/'

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()

        # ego-data process
        ego_output, _ = self.forward(data_dict=cav_content, if_single=True, dataset=dataset)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        # 生成 batch_dict
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

        # fuse-data process
        if num_agent == 1:
            np.save(match_path + save_dir + f'/sample_{num}.npy',[])
            return
        else:
            # attack = attack.item()
            match_loss_list = []
            matcher = HungarianMatcher()
            attack_model = PGD(self.cls_head, self.att_fuse_module, self.reg_head, record_len, t_matrix, self.backbone, self.generate_predicted_boxes)
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
            for agent in range(1, num_agent):
                delete_list = []
                for j in range(1, num_agent):
                    if j != agent:
                        delete_list.append(j)

                # 得到单独与agent合作的结果
                attack_result,_ = attack_model.inference(batch_dict, actual_attack, attack_src, num=num, delete_list=delete_list)

                if self.shrink_flag:
                    attack_result = self.shrink_conv(attack_result)
                
                cls_1 = self.cls_head(attack_result)
                bbox_1 = self.reg_head(attack_result)

                _, bbox_temp_1 = self.generate_predicted_boxes(cls_1, bbox_1)

                att_output_dict = {'cls_preds': cls_1,
                            'reg_preds': bbox_temp_1,
                            'bbox_preds': bbox_1
                            }
                output_dict['ego'] = att_output_dict

                with torch.no_grad():
                    pred_tensor, pred_score, gt_box_tensor, pred_bbox = \
                        dataset.post_process(data_dict, output_dict)
                pred_box = {'box_tensor':pred_bbox, 'score':pred_score}
                match_loss, _, _ = matcher(pred_box, target_box)
                if agent not in attack_src:
                    match_loss_list.append([match_loss, 0])
                else:
                    match_loss_list.append([match_loss, 1])

            if os.path.exists(match_path + save_dir) == False:
                os.makedirs(match_path + save_dir)
            np.save(match_path + save_dir + f'/sample_{num}.npy',match_loss_list)   
            return
    
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
    
    def attack_detection_match(self, data_dict, dataset, attack = None, method = 'match_cost', save_dir = 'test_set_pred', num = 0, attack_type = 'pgd', attack_conf = None):
        """
        inputs:
            data_dict: origin data which includes other agents' information
            attack: (Dict) have 4 parts -- attack_src, block 1, block 2, block 3
            method: detection method like "match_cost","raw autoencoder" and so on
            dataset: used for the data post_process
        """

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        percentiles = match_percentiles
        match_path = 'generate_match/'

        # ego-data process
        ego_output, _ = self.forward(data_dict=cav_content, if_single=True, dataset=dataset)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        # 生成 batch_dict
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

        if not os.path.exists(match_path + save_dir):
            os.makedirs(match_path + save_dir)
        
        # fuse-data process
        # 如果agent = 1直接返回ego的结果即可
        if num_agent == 1:
            np.save(match_path + save_dir + f'/sample_{num}.npy',[])
            return target_box_tensor, target_score, gt_box_tensor, []
        else:
            match_loss_list = []
            if attack_type != 'no attack':
                attack = attack.item()
            matcher = HungarianMatcher()
            attack_model = PGD(self.cls_head, self.att_fuse_module, self.reg_head, record_len, t_matrix, self.backbone, self.generate_predicted_boxes)
            attack_src = attack['src']
            if attack_src == []:
                actual_attack = None
            else:
                actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]

            attack_target = attack_conf.attack.attack_target

            # 生成pred_gt_box_tensor
            no_att_output_dict, _ = self.forward(data_dict=cav_content, dataset=dataset)
            output_dict = OrderedDict()
            output_dict['ego'] = no_att_output_dict
            with torch.no_grad():
                pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
                dataset.post_process(data_dict, output_dict)

            if attack_type == 'erase_and_shift_and_pgd':
                erase_index = self.generate_erase_index(ego_output, data_dict['ego'], dataset, attack_target, attack_conf, pred_gt_box_tensor)

            if 'shift' in attack_type and attack_conf.attack.shift.shift_direction == 'random':
                folder_path_shift = 'outcome/shift_dir_of_box/1013/[1.5, 1.5, 0]'
                shift_dir_of_box = np.load(folder_path_shift + f'sample_{num}.npy')
            else:
                shift_dir_of_box = None

            if attack_type == 'no attack':
                label = 0
            else:
                label = 1


            final_delete_list = []
            for agent in range(1, num_agent):
                # 生成delete list
                delete_list = []
                for j in range(1, num_agent):
                    if j != agent:
                        delete_list.append(j)

                # 得到单独与agent合作的结果
                if attack_type == 'shift_and_pgd':
                    attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box,gt_box_tensor = gt_box_tensor)
                elif attack_type == 'erase_and_shift_and_pgd':
                    attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], if_erase = True, erase_index = erase_index, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box,gt_box_tensor = gt_box_tensor)
                else:
                    attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src, delete_list)

                if self.shrink_flag:
                    attack_result = self.shrink_conv(attack_result)
                
                cls_1 = self.cls_head(attack_result)
                bbox_1 = self.reg_head(attack_result)

                _, bbox_temp_1 = self.generate_predicted_boxes(cls_1, bbox_1)

                att_output_dict = {'cls_preds': cls_1,
                            'reg_preds': bbox_temp_1,
                            'bbox_preds': bbox_1
                            }
                output_dict['ego'] = att_output_dict

                with torch.no_grad():
                    pred_tensor, pred_score, gt_box_tensor, pred_bbox = \
                        dataset.post_process(data_dict, output_dict)
                pred_box = {'box_tensor':pred_bbox, 'score':pred_score}
                match_loss, _, _ = matcher(pred_box, target_box)

                if agent not in attack_src:
                    match_loss_list.append([match_loss, 0])
                else:
                    match_loss_list.append([match_loss, 1])

                if match_loss[0] > percentiles[95]:
                    final_delete_list.append(agent)
                    pred = 1
                else:
                    pred = 0 

            # 存储match_loss结果
            np.save(match_path + save_dir + f'/sample_{num}.npy',match_loss_list)  

            # 得到最终的结果并返回
            if attack_type == 'shift_and_pgd':
                    attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list = final_delete_list, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor)
            elif attack_type == 'erase_and_shift_and_pgd':
                attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list = final_delete_list ,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], if_erase = True, erase_index = erase_index, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor)
            else:
                attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src, delete_list = final_delete_list)

            if self.shrink_flag:
                attack_result = self.shrink_conv(attack_result)
            
            cls_1 = self.cls_head(attack_result)
            bbox_1 = self.reg_head(attack_result)

            _, bbox_temp_1 = self.generate_predicted_boxes(cls_1, bbox_1)

            final_output_dict = {'cls_preds': cls_1,
                        'reg_preds': bbox_temp_1,
                        'bbox_preds': bbox_1
                        }
            output_dict['ego'] = final_output_dict
            
            with torch.no_grad():
                box_tensor, score, gt_box_tensor, bbox = \
                    dataset.post_process(data_dict, output_dict)
            return box_tensor, score, gt_box_tensor, [pred, label]

    
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

    
    def attack_detection_multi_test(self, data_dict, dataset, attack, num, attack_type, attack_conf, save_dir):

        # 多重检验初始化
        dists = []
        dists.append(np.load("generate_match/validation_1016_new/validation_match_cost.npy"))
        dists.append(np.load('generate_ae_loss/residual/validation_1016_new/validation_ae_loss.npy'))
        bh_test = build_bh_procedure(dists=dists)

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        ae_percentiles = residual_ae_percentiles

        # ego-data process
        ego_output, _ = self.forward(data_dict=cav_content, if_single=True, dataset=dataset)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        final_delete_list_path = 'outcome/final_delete_list_v2/' + save_dir
        if not os.path.exists(final_delete_list_path):
            os.makedirs(final_delete_list_path)

        # 生成 batch_dict
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

        # fuse-data process
        if num_agent == 1:
            np.save(final_delete_list_path + f'/sample_{num}.npy', [])
            return target_box_tensor, target_score, gt_box_tensor, []

        match_folder = 'generate_match/' + save_dir
        ae_type = 'residual'
        ae_folder = 'generate_ae_loss/' + ae_type + '/' + save_dir
        match_loss = np.load(match_folder + f'/sample_{num}.npy', allow_pickle=True)
        ae_loss = np.load(ae_folder + f'/sample_{num}.npy', allow_pickle=True)

        # 生成结果
        final_delete_list = []
        scores = [match_loss[0][0][0].cpu().numpy(), ae_loss[0][0][0]]
        rejected = bh_test.test_v2(scores)
        
        if rejected > 0:
            final_delete_list.append(1)
            pred = 1
        else:
            pred = 0

        attack_target = attack_conf.attack.attack_target

        # 生成pred_gt_box_tensor
        no_att_output_dict, _ = self.forward(data_dict=cav_content, dataset=dataset)
        output_dict = OrderedDict()
        output_dict['ego'] = no_att_output_dict
        with torch.no_grad():
            pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
            dataset.post_process(data_dict, output_dict)

        # About Erase And Shift
        if attack_type == 'erase_and_shift_and_pgd':
            erase_index = self.generate_erase_index(ego_output, data_dict['ego'], dataset, attack_target, attack_conf, pred_gt_box_tensor)

        if 'shift' in attack_type and attack_conf.attack.shift.shift_direction == 'random':
            folder_path_shift = 'outcome/shift_dir_of_box/1013/[1.5, 1.5, 0]'
            shift_dir_of_box = np.load(folder_path_shift + f'sample_{num}.npy')
        else:
            shift_dir_of_box = None
        
        # 存储结果
        np.save(final_delete_list_path + f'/sample_{num}.npy', final_delete_list)

        attack_model = PGD(self.cls_head, self.att_fuse_module, self.reg_head, record_len, t_matrix, self.backbone, self.generate_predicted_boxes)

        # About Attack
        if attack_type != 'no attack':
            attack = attack.item()
            label = 1
        else:
            label = 0

        attack_src = attack['src']
        if attack_src == []:
            actual_attack = None
        else:
            actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
        attack_target = attack_conf.attack.attack_target

        if attack_type == 'shift_and_pgd':
            attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list = final_delete_list, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor)
        elif attack_type == 'erase_and_shift_and_pgd':
            attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list = final_delete_list ,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], if_erase = True, erase_index = erase_index, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor)
        else:
            attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src, delete_list = final_delete_list)

        if self.shrink_flag:
            attack_result = self.shrink_conv(attack_result)
        cls_1 = self.cls_head(attack_result)
        bbox_1 = self.reg_head(attack_result)
        _, bbox_temp_1 = self.generate_predicted_boxes(cls_1, bbox_1)
        final_output_dict = {'cls_preds': cls_1,
                    'reg_preds': bbox_temp_1,
                    'bbox_preds': bbox_1
                    }
        output_dict['ego'] = final_output_dict
        
        with torch.no_grad():
            box_tensor, score, gt_box_tensor, bbox = \
                dataset.post_process(data_dict, output_dict)
        return box_tensor, score, gt_box_tensor, [pred, label]


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

                        # if if_inference:
                        #     # self.visualize_tensor_both(attacker_normal, x[attacker], num, min_x, max_x, attack_conf, i)
                        #     self.visualize_tensor_single(attacker_normal, x[attacker], num, min_x, max_x, attack_conf, i)

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
    def forward(self, data_dict, attack=False, attack_target='pred', save_path=None, num = 0, save_attack = False, delete_list = [], attack_type = 'pgd', dataset = None, if_single = False, save_dir = None):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        #  

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
        if attack == False:
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
        
        # if num > 30:
        #     import pdb; pdb.set_trace()
        

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

                # shift_dir_save_path = f'outcome/shift_dir_of_box/1013/{attack_conf.attack.pgd.eps}'
                # shift_dir_of_box = np.load(shift_dir_save_path + f'sample_{num}.npy')
                # if len(shift_dir_of_box) - len(shift_dir_of_box_1) != 0:
                #     print(len(shift_dir_of_box) - len(shift_dir_of_box_1))
                #     np.save(shift_dir_save_path + f'sample_{num}.npy', shift_dir_of_box)
                # else:
                #     tmp_dir_box = [b]

                shift_dir_save_path = f'outcome/shift_dir_of_box/1216/{attack_conf.attack.pgd.eps}/'
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
            elif attack_type == 'rotate_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = False, rotate_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, no_att_output_dict = no_att_output_dict)
            elif attack_type == 'rotate':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs)
            elif attack_type == 'shift_and_rotate':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs)
            # 

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
                elif attack_type == 'rotate_and_pgd':
                    tmp_save_path = save_path + f"_{attack_conf.attack.rotate.bbox_num}bbox_{attack_conf.attack.rotate.shift_angle}angle_eps{attack_conf.attack.pgd.eps[0]}"
                    if not os.path.exists(tmp_save_path):
                        os.makedirs(tmp_save_path)
                    np.save(tmp_save_path + f'/sample_{num}.npy', tmp)
                elif attack_type == 'rotate':
                    if not os.path.exists(save_path + f"_{attack_conf.attack.rotate.bbox_num}bbox_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}"):
                        os.makedirs(save_path + f"_{attack_conf.attack.rotate.bbox_num}bbox_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}")
                    np.save(save_path + f"_{attack_conf.attack.rotate.bbox_num}bbox_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}" + f'/sample_{num}.npy', tmp)
                elif attack_type == 'shift_and_rotate':
                    if not os.path.exists(save_path + f"_{attack_conf.attack.bbox_num}bbox_shift_{attack_conf.attack.shift.shift_length}_{attack_conf.attack.shift.shift_direction}_rotate_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}"):
                        os.makedirs(save_path + f"_{attack_conf.attack.bbox_num}bbox_shift_{attack_conf.attack.shift.shift_length}_{attack_conf.attack.shift.shift_direction}_rotate_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}")
                    np.save(save_path + f"_{attack_conf.attack.bbox_num}bbox_shift_{attack_conf.attack.shift.shift_length}_{attack_conf.attack.shift.shift_direction}_rotate_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}" + f'/sample_{num}.npy', tmp)

            # import ipdb; ipdb.set_trace
            if attack_type == 'shift_and_pgd':
                fused_feature_1, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor)
            elif attack_type == 'erase_and_shift_and_pgd':
                fused_feature_1, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, if_erase = True, erase_index = erase_index, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor, cls = cls)
            elif attack_type == 'rotate_and_pgd':
                fused_feature_1, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list, shift_feature = False, rotate_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_target = attack_target, no_att_output_dict = no_att_output_dict, dataset = dataset)
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