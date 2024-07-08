import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
import torch


from collections import OrderedDict
from opencood.utils.match import HungarianMatcher
from opencood.utils.residual_autoencoder import ResidualAutoEncoderV2ReconstructionLoss
from opencood.utils.bh_procedure import build_bh_procedure
from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
from opencood.utils import eval_utils
from .pgd import PGD
from .shift import Shift
from .rotate import Rotate
from .shift_and_rotate import Shift_and_Rotate
import os 
import time

class CenterPointWhere2comm(nn.Module):
    def __init__(self, args):
        super(CenterPointWhere2comm, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        
        self.voxel_size = args['voxel_size']
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range  = args['lidar_range']

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        
        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 8 * args['anchor_number'],
                                  kernel_size=1)
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
    
    # Not Used
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    # Generate attack sources randomly
    def get_attack_src(self, agent_num, n_att):

        if agent_num - 1 < n_att:
            return []

        attacker = torch.randint(low=1,high=agent_num,size=(n_att,))
        tmp = []
        for i in range(len(attacker)):
            tmp.append(attacker[i])

        return tmp

    # TODO: other parameters of attack will be updated
    def forward(self, data_dict, attack=False, attack_target='pred', save_path=None, num = 0, save_attack = False, delete_list = [], attack_type = 'pgd', dataset = None, if_single = False):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict) # spatial_features: (N, 64, 200, 504)
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # print('spatial_features_2d: ', spatial_features_2d.shape)


        if self.multi_scale:
            fused_feature, communication_rates, result_dict, _ = self.fusion_net(batch_dict['spatial_features'],
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone, data_dict = batch_dict, num = num)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict, _ = self.fusion_net(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, data_dict = batch_dict, num = num)
            
            
        # print('fused_feature: ', fused_feature.shape)
        cls = self.cls_head(fused_feature)
        bbox = self.reg_head(fused_feature)

        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)

        no_att_output_dict = {'cls_preds': cls,
                       'reg_preds': bbox_temp,
                       'bbox_preds': bbox
                       }
        no_att_output_dict.update(result_dict)

        _, bbox_temp_single = self.generate_predicted_boxes(psm_single, rm_single)

        no_att_output_dict.update({'cls_preds_single': psm_single,
                       'reg_preds_single': bbox_temp_single,
                       'bbox_preds_single': rm_single,
                       'comm_rate': communication_rates
                       })
        
        # Generate pred outcome
        output_dict = OrderedDict()
        output_dict['ego'] = no_att_output_dict
        tmp_data_dict = OrderedDict()
        tmp_data_dict['ego'] = data_dict
        with torch.no_grad():
            pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
                dataset.post_process(tmp_data_dict, output_dict)
            
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
                    self.attack_model = PGD(self.fusion_net, self.cls_head, self.reg_head, record_len, 
                                            pairwise_t_matrix, self.backbone, self.generate_predicted_boxes,**attack_conf.attack.pgd)
                    self.attack_target = attack_conf.attack.attack_target
                    n_att = attack_conf.attack.pgd.n_att
                    attack_srcs = self.get_attack_src(batch_dict['spatial_features'].shape[0], n_att)
                else:
                    self.attack = False
            elif attack_type == 'shift':
                if attack_conf.attack is not None:
                    self.attack = True
                    self.attack_model = Shift(self.fusion_net, **attack_conf.attack.shift)
                    self.attack_target = attack_conf.attack.shift.attack_target
                    n_att = attack_conf.attack.shift.n_att
                    attack_srcs = self.get_attack_src(batch_dict['spatial_features'].shape[0], n_att)
                else:
                    self.attack = False
            elif attack_type == 'erase_and_shift':
                if attack_conf.attack is not None:
                    self.attack = True
                    self.attack_model = Shift(self.fusion_net, **attack_conf.attack.shift)
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
                    # att_reg_targets = ref_result['reg_preds'].permute(0, 2, 3, 1).contiguous()
                    att_labels = ref_result['cls_preds'].permute(0, 2, 3, 1).contiguous()
                else:
                    raise NotImplementedError(self.attack_target)

            # If erasion exists, Generate Erase Index
            if attack_type == 'erase_and_shift_and_pgd' or attack_type == 'erase_and_shift':

                without_attacker_output_dict = {'cls_preds': psm_single[0].unsqueeze(0),
                'reg_preds': rm_single[0].unsqueeze(0)}
                
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
                
                # print(erase_index)
            
            # If shift, Generate shift directions
            if attack_type == 'erase_and_shift_and_pgd' and attack_conf.attack.shift.shift_direction == 'random':
                if attack_target == 'gt':
                    bbox_num = gt_box_tensor.shape[0] - len(erase_index)
                else:
                    bbox_num = pred_gt_box_tensor.shape[0] - len(erase_index)
                shift_dir_of_box = [np.random.randint(low=0,high=4) for k in range(bbox_num)]
            elif attack_type == 'shift_and_pgd' and attack_conf.attack.shift.shift_direction == 'random':
                if attack_target == 'gt':
                    bbox_num = gt_box_tensor.shape[0]
                else:
                    bbox_num = pred_gt_box_tensor.shape[0]
                shift_dir_of_box = [np.random.randint(low=0,high=4) for k in range(bbox_num)]
            elif attack_type == 'shift' and attack_conf.attack.shift.shift_direction == 'random':
                if attack_target == 'gt':
                    bbox_num = gt_box_tensor.shape[0]
                else:
                    bbox_num = pred_gt_box_tensor.shape[0]
                shift_dir_of_box = [np.random.randint(low=0,high=4) for k in range(bbox_num)]

            if attack_type == 'pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, attack_srcs = attack_srcs)
            elif attack_type == 'shift':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs, attack_target = attack_target, no_att_output_dict = no_att_output_dict, dataset = dataset, shift_dir_of_box = shift_dir_of_box)
            elif attack_type == 'shift_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, no_att_output_dict = no_att_output_dict, shift_dir_of_box = shift_dir_of_box, dataset = dataset)
            elif attack_type == 'erase_and_shift_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, if_erase = True, erase_index = erase_index, dataset = dataset, no_att_output_dict = no_att_output_dict, shift_dir_of_box = shift_dir_of_box)
            elif attack_type == 'erase_and_shift':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs, if_erase = True, erase_index = erase_index, num = num, attack_conf = attack_conf, attack_target = attack_target, no_att_output_dict = no_att_output_dict, dataset = dataset)
            elif attack_type == 'rotate_and_pgd':
                evasion, attack_src = self.attack_model(batch_dict, anchors, att_reg_targets, att_labels, num = num, attack_target = self.attack_target, shift_feature = False, rotate_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_srcs = attack_srcs, no_att_output_dict = no_att_output_dict)
            elif attack_type == 'rotate':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs)
            elif attack_type == 'shift_and_rotate':
                evasion, attack_src = self.attack_model(data_dict, batch_dict, attack_srcs = attack_srcs)

            # Save attack 
            if save_attack:

                # TODO: attack的形状
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

            if attack_type == 'shift_and_pgd':
                fused_feature_1, communication_rates, result_dict, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_target = attack_target, no_att_output_dict = no_att_output_dict, dataset = dataset, shift_dir_of_box=shift_dir_of_box)
            elif attack_type == 'erase_and_shift_and_pgd':
                fused_feature, communication_rates, result_dict, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, if_erase = True, erase_index = erase_index, attack_target = attack_target, no_att_output_dict = no_att_output_dict, dataset = dataset, shift_dir_of_box=shift_dir_of_box)
            elif attack_type == 'rotate_and_pgd':
                fused_feature, communication_rates, result_dict, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num,delete_list=delete_list, shift_feature = False, rotate_feature = True, attack_conf = attack_conf, real_data_dict = data_dict, attack_target = attack_target, no_att_output_dict = no_att_output_dict, dataset = dataset)
            else:
                fused_feature_1, communication_rates, result_dict, _ = self.attack_model.inference(batch_dict, evasion, attack_src=attack_src, num = num, delete_list=delete_list)

            if self.shrink_flag:
                fused_feature_1 = self.shrink_conv(fused_feature_1)
            
            cls_1 = self.cls_head(fused_feature_1)
            bbox_1 = self.reg_head(fused_feature_1)

            _, bbox_temp_1 = self.generate_predicted_boxes(cls_1, bbox_1)

            att_output_dict = {'cls_preds': cls_1,
                        'reg_preds': bbox_temp_1,
                        'bbox_preds': bbox_1
                        }
            att_output_dict.update(result_dict)   

            _, bbox_temp_single = self.generate_predicted_boxes(psm_single, rm_single)

            att_output_dict.update({'cls_preds_single': psm_single,
                        'reg_preds_single': bbox_temp_single,
                        'bbox_preds_single': rm_single,
                        'comm_rate': communication_rates
                        })
        else:
            att_output_dict = {'cls_preds': None, 'reg_preds': None,
            'bbox_preds': None}
        
        # TODO: Save Erase Index
        # if attack_type == 'erase_and_shift':
        #     output_dict = OrderedDict()
        #     output_dict['ego'] = att_output_dict
        #     tmp_data_dict = OrderedDict()
        #     tmp_data_dict['ego'] = data_dict
        #     with torch.no_grad():
        #         box_tensor, pred_score, gt_box_tensor, bbox = \
        #             dataset.post_process(tmp_data_dict, output_dict)
            
        #     if attack_target == 'gt':
        #         erase_fail_index = eval_utils.find_box_erase_fail(box_tensor, score, gt_box_tensor[erase_index], 0.3, erase_index)
        #     else:
        #         erase_fail_index = eval_utils.find_box_erase_fail(box_tensor, score, pred_gt_box_tensor[erase_index], 0.3, erase_index)
        #     np.save(f'/GPFS/data/shengyin/OpenCOOD-main/outcome/erase_index_without_pgd/success/sample{num}', erase_index)
        #     np.save(f'/GPFS/data/shengyin/OpenCOOD-main/outcome/erase_index_without_pgd/fail/sample{num}', erase_fail_index)
        
        if self.attack and (attack_type == 'erase_and_shift' or attack_type == 'erase_and_shift_and_pgd'):
            return att_output_dict, erase_index
        elif self.attack:
            return att_output_dict, None
        else:
            if if_single:
                no_att_output_dict['cls_preds'] = no_att_output_dict['cls_preds_single']
                no_att_output_dict['reg_preds'] = no_att_output_dict['reg_preds_single']
                no_att_output_dict['bbox_preds'] = no_att_output_dict['bbox_preds_single']
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