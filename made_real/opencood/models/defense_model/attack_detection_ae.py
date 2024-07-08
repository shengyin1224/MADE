from collections import OrderedDict
import numpy as np
import torch
import sys
import random
import os

from opencood.models.center_point_baseline_multiscale import CenterPointBaselineMultiscale
from config.defense.percentiles import *
from opencood.utils.match import HungarianMatcher
from opencood.models.multiscale_attn_pgd import PGD

class AttackDetectionAe():
    
    def __init__(self, model):
        
        self.Model = model
    
    def forward(self, data_dict, dataset, attack = None, method = 'residual', save_dir = 'test_set_pred', num = 0, attack_type = 'pgd', attack_conf = None, defense_model = None):

        ae_type = method
        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        ae_path = 'generate_ae_loss/' + ae_type + '/'

        # ego-data process
        ego_output, _ = self.Model.forward(data_dict=cav_content, if_single=True, dataset=dataset)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)

        # generate batch dict
        batch_dict, t_matrix, record_len = self.Model.generate_batch_dict(cav_content)
        feature_list = batch_dict['feature_list']

        if not os.path.exists(ae_path + save_dir):
            os.makedirs(ae_path + save_dir)
        
        # fuse-data process
        if num_agent == 1 or attack == None:
            np.save(ae_path + save_dir + f'/sample_{num}.npy',[])
            return target_box_tensor, target_score, gt_box_tensor, []

        if attack_type != 'no_attack':
            attack = attack.item()
        attack_src = attack['src']
        if attack_src == []:
            actual_attack = None
        else:
            actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
        
        if attack_conf != None:
            attack_target = attack_conf.attack.attack_target
        else:
            attack_target = 'pred'

        # 生成pred_gt_box_tensor
        no_att_output_dict, _ = self.Model.forward(data_dict=cav_content, dataset=dataset)
        output_dict = OrderedDict()
        output_dict['ego'] = no_att_output_dict
        with torch.no_grad():
            pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
            dataset.post_process(data_dict, output_dict)
        
        if attack_type == 'erase_and_shift_and_pgd':
            erase_index = self.Model.generate_erase_index(ego_output, data_dict['ego'], dataset, attack_target, attack_conf, pred_gt_box_tensor)
        else:
            erase_index = None

        if 'shift' in attack_type and attack_conf.attack.shift.shift_direction == 'random':
            # folder_path_shift = 'outcome/shift_dir_of_box/1013/[1.5, 1.5, 0]'
            folder_path_shift = 'outcome/shift_dir_of_box/1202_train/[1.5, 1.5, 0]'
            shift_dir_of_box = np.load(folder_path_shift + f'sample_{num}.npy')
        else:
            shift_dir_of_box = None

        attacked_feature_list = self.Model.apply_attack(feature_list, actual_attack, attack_src, attack_type, record_len, t_matrix, attack_conf, data_dict['ego'], batch_dict, erase_index, num, attack_target, pred_gt_box_tensor, dataset, shift_dir_of_box, gt_box_tensor)

        if attack_type == 'no_attack': 
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
            for i, fuse_module in enumerate(self.Model.fusion_net):
                fused_feature_list.append(fuse_module(attacked_feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
                no_fused_feature_list.append(fuse_module(add_zero_feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
            
            fused_feature = self.Model.backbone.decode_multiscale_feature(fused_feature_list).squeeze(0) 
            no_fused_feature = self.Model.backbone.decode_multiscale_feature(no_fused_feature_list).squeeze(0)

            residual_feature = (fused_feature - no_fused_feature).unsqueeze(0)
            
            # residual_feature = (fused_feature_list[0] - no_fused_feature_list[0
            # ])

            # centralize
            # mean_feature = torch.from_numpy(np.load('opencood/ae_train/1010-AE-train/layer0/layer_0.npy')).cuda()
            mean_feature = torch.from_numpy(np.load('generate_ae_loss/residual/mean_feat/1016_res_mean_feature.npy')).cuda()
            residual_feature = residual_feature - mean_feature.unsqueeze(0)

            # vis_feature
            # import pdb; pdb.set_trace()
            # overall_path = 'outcome/vis_residual_1216/' + 'no_attack_temp20_'
            # residual_vis_path = overall_path + 'after_decoder'
            # if not os.path.exists(residual_vis_path):
            #     os.makedirs(residual_vis_path)
            # residual_vis_feature = residual_feature.clone().detach().cpu().numpy()
            # plt.imshow(residual_vis_feature.squeeze(0).mean(axis=0))
            # plt.colorbar()
            # plt.savefig(residual_vis_path + f'/sample_{num}.png')
            # plt.close()

            # residual_vis_path = overall_path + 'layer0'
            # if not os.path.exists(residual_vis_path):
            #     os.makedirs(residual_vis_path)
            # residual_vis_feature = residual_feat_layer0.clone().detach().cpu().numpy()
            # plt.imshow(residual_vis_feature.squeeze(0).mean(axis=0))
            # plt.colorbar()
            # plt.savefig(residual_vis_path + f'/sample_{num}.png')
            # plt.close()

            # residual_vis_path = overall_path + 'layer1'
            # if not os.path.exists(residual_vis_path):
            #     os.makedirs(residual_vis_path)
            # residual_vis_feature = residual_feat_layer1.clone().detach().cpu().numpy()
            # plt.imshow(residual_vis_feature.squeeze(0).mean(axis=0))
            # plt.colorbar()
            # plt.savefig(residual_vis_path + f'/sample_{num}.png')
            # plt.close()

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
        
        if attack_type == 'no_attack':
            np.save(ae_path + save_dir + f'/sample_{num}.npy',np.array([[save_score, 0]], dtype = object))
            label = 0
        else:
            np.save(ae_path + save_dir + f'/sample_{num}.npy',np.array([[save_score, 1]], dtype = object))
            label = 1
        
        new_feature_list, new_record_len, new_t_matrix = self.defense_model_0.del_attacker(id_keep, attacked_feature_list, record_len, t_matrix)

        fused_att_feature_list = []
        for i, fuse_module in enumerate(self.Model.fusion_net):
            fused_att_feature_list.append(fuse_module(new_feature_list[i], new_record_len, torch.stack(new_t_matrix)))
        
        fused_att_feature = self.Model.backbone.decode_multiscale_feature(fused_att_feature_list)

        fused_feature_1 = fused_att_feature
        if self.Model.shrink_flag:
            fused_feature_1 = self.Model.shrink_conv(fused_feature_1)
            
        cls_1 = self.Model.cls_head(fused_feature_1)
        bbox_1 = self.Model.reg_head(fused_feature_1)
        _, bbox_temp_1 = self.Model.generate_predicted_boxes(cls_1, bbox_1)

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