from collections import OrderedDict
import numpy as np
import torch
import sys
import random
import os

from opencood.models.center_point_baseline_multiscale import CenterPointBaselineMultiscale

class generate_ae_train():
    
    def __init__(self, model):
        
        self.Model = model
    
    def forward(self, data_dict, save_dir = 'validation', num = 0, ae_type = 'residual', attack = None, attack_type = 'pgd', attack_conf = None, dataset = None):
        
        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        ae_path = 'generate_ae_loss/' + ae_type + '/' + save_dir
        
        if num_agent == 1 or attack == None:
            return

        # ego-data process
        ego_output, _ = self.Model.forward(data_dict=cav_content, if_single=True, dataset=dataset)

        # generate batch dict
        batch_dict, t_matrix, record_len = self.Model.generate_batch_dict(cav_content)

        # load attack
        if attack_type != 'no_attack':
            attack = attack.item()
        attack_src = attack['src']
        if attack_src == []:
            actual_attack = None
        else:
            actual_attack = [torch.tensor(attack[f'block{i}']).cuda() for i in range(1,4)]
        attack_target = attack_conf.attack.attack_target

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
            if attack_type == 'erase_and_shift_and_pgd':
                bbox_num = pred_gt_box_tensor.shape[0] - len(erase_index)
            else:
                bbox_num = pred_gt_box_tensor.shape[0]
            shift_dir_of_box = [np.random.randint(low=0,high=4) for k in range(bbox_num)]
        else:
            shift_dir_of_box = None

        attacked_feature_list = self.Model.apply_attack(feature_list, actual_attack, attack_src, attack_type, record_len, t_matrix, attack_conf, data_dict['ego'], batch_dict, erase_index, num, attack_target, pred_gt_box_tensor, dataset, shift_dir_of_box, gt_box_tensor)

        feature_list = attacked_feature_list

        if ae_type == 'raw':

            for k in range(3):
                if not os.path.exists(ae_path + f'/layer_{k}'):
                    os.makedirs(ae_path + f'/layer_{k}')

                save_feature = feature_list[k][1].detach().cpu().numpy()
                np.save(ae_path + f'/layer_{k}/sample_{num}.npy',save_feature)
                
        else:

            add_zero_feature_list = [j.clone() for j in feature_list]
            for i in range(3):
                add_zero_feature_list[i][1] = torch.zeros_like(add_zero_feature_list[i][1], device='cuda')
            fused_feature_list = []
            no_fused_feature_list = []
            for i, fuse_module in enumerate(self.Model.fusion_net):
                fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
                no_fused_feature_list.append(fuse_module(add_zero_feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))

            fused_feature = self.Model.backbone.decode_multiscale_feature(fused_feature_list).squeeze(0) 
            no_fused_feature = self.Model.backbone.decode_multiscale_feature(no_fused_feature_list).squeeze(0)

            save_feature = (fused_feature - no_fused_feature).detach().cpu().numpy()

            if os.path.exists(ae_path) == False:
                os.makedirs(ae_path)

            np.savez(ae_path + f'/sample_{num}.npz', save_feature = save_feature)

        return