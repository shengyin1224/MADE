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

class generate_match():
    
    def __init__(self, model):
        
        self.Model = model
        
    def forward(self, data_dict, dataset, attack = None, num = 0, save_dir = 'validation', match_para = 1):
        
        match_path = 'generate_match/'

        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()

        # ego-data process
        ego_output, _ = self.Model.forward(data_dict=cav_content, if_single=True, dataset=dataset)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}

        # generate batch dict
        batch_dict, t_matrix, record_len = self.Model.generate_batch_dict(cav_content)

        # fuse-data process
        if num_agent == 1 or attack == None:
            np.save(match_path + save_dir + f'/sample_{num}.npy',[])
            return
        else:
            # attack = attack.item()
            match_loss_list = []
            matcher = HungarianMatcher(cost_giou = match_para)
            attack_model = PGD(self.Model.cls_head, self.Model.att_fuse_module, self.Model.reg_head, record_len, t_matrix, self.Model.backbone, self.Model.generate_predicted_boxes)
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

            if os.path.exists(match_path + save_dir) == False:
                os.makedirs(match_path + save_dir)
            np.save(match_path + save_dir + f'/sample_{num}.npy',match_loss_list)   
            return