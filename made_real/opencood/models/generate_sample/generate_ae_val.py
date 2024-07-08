from collections import OrderedDict
import numpy as np
import torch
import sys
import random
import os

from opencood.models.center_point_baseline_multiscale import CenterPointBaselineMultiscale
from config.defense.percentiles import *

class generate_ae_val():
    
    def __init__(self, model):
        
        self.Model = model
    
    def forward(self, data_dict, save_dir = 'validation', num = 0, ae_type = 'residual', attack = None, attack_type = 'pgd', attack_conf = None, dataset = None, defense_model = None):
        
        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        ae_path = 'generate_ae_loss/' + ae_type + '/' + save_dir

        if num_agent == 1 or attack == None:
            return

        # generate batch dict
        batch_dict, t_matrix, record_len = self.Model.generate_batch_dict(cav_content)

        if not os.path.exists(ae_path):
            os.makedirs(ae_path)
        
        # fuse-data process
        feature_list = batch_dict['feature_list']
        attacked_feature_list = batch_dict['feature_list']

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
            for i, fuse_module in enumerate(self.Model.fusion_net):
                fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None))
                no_fused_feature_list.append(fuse_module(add_zero_feature_list[i], record_len, t_matrix, num = num, if_draw = False, cls = None)) 

                    
            fused_feature = self.Model.backbone.decode_multiscale_feature(fused_feature_list).squeeze(0) 
            no_fused_feature = self.Model.backbone.decode_multiscale_feature(no_fused_feature_list).squeeze(0)

            residual_feature = (fused_feature - no_fused_feature)

            # centralize
            mean_feature = torch.from_numpy(np.load('generate_ae_loss/residual/mean_feat/1016_res_mean_feature.npy')).cuda()
            residual_feature = residual_feature - mean_feature.unsqueeze(0)

            self.defense_model = defense_model

            id_keep, additional_dict = self.defense_model(
                                        residual_feature,
                                        record_len,
                                        t_matrix,
                                        [torch.arange(1,2).cuda()],
                                        [torch.ones(0).cuda()])
                                        
            score = [additional_dict['score']]

            np.save(ae_path + f'/sample_{num}.npy', score)

        return