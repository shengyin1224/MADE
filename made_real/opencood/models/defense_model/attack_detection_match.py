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
    
class AttackDetectionMatch():
    
    def __init__(self, model):
        
        self.Model = model
    
    def forward(self, data_dict, dataset, attack = None, save_dir = 'test_set_pred', num = 0, attack_type = 'pgd', attack_conf = None, match_para = 1):
        
        # import pdb; pdb.set_trace()
        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        percentiles = match_percentiles
        match_path = 'generate_match/'

        # ego-data process
        ego_output, _ = self.Model.forward(data_dict=cav_content, if_single=True, dataset=dataset)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        # generate batch dict
        batch_dict, t_matrix, record_len = self.Model.generate_batch_dict(cav_content)

        if not os.path.exists(match_path + save_dir):
            os.makedirs(match_path + save_dir)
        
        # fuse-data process
        if num_agent == 1 or attack == None:
            np.save(match_path + save_dir + f'/sample_{num}.npy',[])
            return target_box_tensor, target_score, gt_box_tensor, []
        else:
            match_loss_list = []
            if attack_type != 'no_attack':
                attack = attack.item()
            matcher = HungarianMatcher(cost_giou=match_para)
            attack_model = PGD(self.Model.cls_head, self.Model.att_fuse_module, self.Model.reg_head, record_len, t_matrix, self.Model.backbone, self.Model.generate_predicted_boxes)
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

            if 'shift' in attack_type and attack_conf.attack.shift.shift_direction == 'random':
                folder_path_shift = 'outcome/shift_dir_of_box/1013/[1.5, 1.5, 0]'
                shift_dir_of_box = np.load(folder_path_shift + f'sample_{num}.npy')
            else:
                shift_dir_of_box = None

            if attack_type == 'no_attack':
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

                if self.Model.shrink_flag:
                    attack_result = self.Model.shrink_conv(attack_result)
                
                cls_1 = self.Model.cls_head(attack_result)
                bbox_1 = self.Model.reg_head(attack_result)

                _, bbox_temp_1 = self.Model.generate_predicted_boxes(cls_1, bbox_1)

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

                if match_loss[0] > percentiles[match_para]:
                    final_delete_list.append(agent)
                    pred = 1
                else:
                    pred = 0 

            np.save(match_path + save_dir + f'/sample_{num}.npy',match_loss_list)  

            if attack_type == 'shift_and_pgd':
                    attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list = final_delete_list, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor)
            elif attack_type == 'erase_and_shift_and_pgd':
                attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src,delete_list = final_delete_list ,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], if_erase = True, erase_index = erase_index, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box, gt_box_tensor = gt_box_tensor)
            else:
                attack_result, _ = attack_model.inference(batch_dict, actual_attack, attack_src, delete_list = final_delete_list)

            if self.Model.shrink_flag:
                attack_result = self.Model.shrink_conv(attack_result)
            
            cls_1 = self.Model.cls_head(attack_result)
            bbox_1 = self.Model.reg_head(attack_result)

            _, bbox_temp_1 = self.Model.generate_predicted_boxes(cls_1, bbox_1)

            final_output_dict = {'cls_preds': cls_1,
                        'reg_preds': bbox_temp_1,
                        'bbox_preds': bbox_1
                        }
            output_dict['ego'] = final_output_dict
            
            with torch.no_grad():
                box_tensor, score, gt_box_tensor, bbox = \
                    dataset.post_process(data_dict, output_dict)
            return box_tensor, score, gt_box_tensor, [pred, label]