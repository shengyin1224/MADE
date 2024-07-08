from collections import OrderedDict
import numpy as np
import torch
import sys
import random
import os

from opencood.models.center_point_baseline_multiscale import CenterPointBaselineMultiscale
from opencood.models.multiscale_attn_pgd import PGD
from scipy.optimize import linear_sum_assignment
sys.path.append('/GPFS/data/shengyin/DAMC-HPC/Rotated_IoU')
from oriented_iou_loss import cal_giou_3d, cal_iou_3d

class ROBOSAC():
    
    def __init__(self, model, rob_type = 'robosac_mAP', robosac_k = None, step_budget = 3, match_thresh = 0.3, use_history = False, estimate_att_ratio = [0.0, 0.2, 0.4, 0.6, 0.8]):
        
        '''
        Paras:
        
            rob_type: robosac_mAP/adaptive/fix_attackers
            robosac_k: specify consensus set size if needed
            step_budget: sampling budget in a single frame
            match_thresh: IoU threshold for validating two detection results
            use_history: use history frame for computing the consensus, reduce 1 step of forward prop
            
        '''
        # super(ROBOSAC, self).__init__()
        
        self.rob_type = rob_type
        self.robosac_k = robosac_k
        self.step_budget = step_budget
        self.match_thresh = match_thresh
        self.use_history = use_history
        self.estimate_att_ratio = estimate_att_ratio
        
        self.Model = model
        
    
    def cal_robosac_consensus(self, num_agent, step_budget, num_attackers):
        num_agent = num_agent - 1
        eta = num_attackers / num_agent
        s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
        return s
    
    def cal_robosac_steps(self, num_agent, num_consensus, num_attackers):
        num_agent = num_agent - 1
        eta = num_attackers / num_agent
        N = np.ceil(np.log(1 - 0.99) / np.log(1 - np.power(1 - eta, num_consensus))).astype(int)
        return N
        
    def run_attack(self, data_dict, dataset, attack_type, attack_target, attack_conf, pred_gt_box_tensor, ego_output, num, batch_dict, actual_attack, attack_src, gt_box_tensor, collab_agent_list): 
        
        if attack_type == 'erase_and_shift_and_pgd':
            erase_index = self.generate_erase_index(ego_output, data_dict['ego'], dataset, attack_target, attack_conf, pred_gt_box_tensor)

        if 'shift' in attack_type and attack_conf.attack.shift.shift_direction == 'random':
            folder_path_shift = 'outcome/shift_dir_of_box/1013/[1.5, 1.5, 0]'
            shift_dir_of_box = np.load(folder_path_shift + f'sample_{num}.npy')
        else:
            shift_dir_of_box = None

        if attack_type == 'shift_and_pgd':
            attack_result, _ = self.attack_model.inference(batch_dict, actual_attack, attack_src,collab_agent_list = collab_agent_list, shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box,gt_box_tensor = gt_box_tensor)
        elif attack_type == 'erase_and_shift_and_pgd':
            attack_result, _ = self.attack_model.inference(batch_dict, actual_attack, attack_src,collab_agent_list = collab_agent_list,shift_feature = True, attack_conf = attack_conf, real_data_dict = data_dict['ego'], if_erase = True, erase_index = erase_index, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box=shift_dir_of_box,gt_box_tensor = gt_box_tensor)
        else:
            attack_result, _ = self.attack_model.inference(batch_dict, actual_attack, attack_src, collab_agent_list = collab_agent_list)

        if self.Model.shrink_flag:
            attack_result = self.Model.shrink_conv(attack_result)
            
        cls_1 = self.Model.cls_head(attack_result)
        bbox_1 = self.Model.reg_head(attack_result)

        _, bbox_temp_1 = self.Model.generate_predicted_boxes(cls_1, bbox_1)

        att_output_dict = {'cls_preds': cls_1,
                    'reg_preds': bbox_temp_1,
                    'bbox_preds': bbox_1
                    }
        output_dict = OrderedDict()
        output_dict['ego'] = att_output_dict
        with torch.no_grad():
            pred_tensor, pred_score, gt_box_tensor, pred_bbox = \
                dataset.post_process(data_dict, output_dict)
        pred_box = {'box_tensor':pred_bbox, 'score':pred_score}
        
        return att_output_dict, pred_tensor, pred_score, pred_box
    
    def get_jaccard_index(self, pred_boxes, target_boxes):

        match_cost = []
        
        if pred_boxes['box_tensor'] == None:
            m = 0
        else:
            m = pred_boxes['box_tensor'].shape[0]
            if m > 50:
                m = 50
        if target_boxes['box_tensor'] == None:
            n = 0
        else:
            n = target_boxes['box_tensor'].shape[0]
        cost_mat = torch.zeros((max(m, n), max(m, n)))  
        box_cost = torch.zeros((max(m, n), max(m, n)))
        
        if m > n:
            box_cost[:, n:] = 0
        elif m < n:
            box_cost[m:, :] = 1.0
        
        for i in range(m):
            for j in range(n):

                giou_loss, iou3d = cal_giou_3d(pred_boxes['box_tensor'][i].unsqueeze(0).unsqueeze(0), target_boxes['box_tensor'][j].unsqueeze(0).unsqueeze(0))
                # ious = cal_iou_3d(pred_boxes['box_tensor'][i].unsqueeze(0).unsqueeze(0), target_boxes['box_tensor'][j].unsqueeze(0).unsqueeze(0), verbose=True)
                box_cost[i, j] = 1.0 - iou3d.squeeze(0)

        # TODO: iou
        cost_mat = box_cost

        # 完成bbox一一匹配
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        match_cost.append(1 - cost_mat[row_ind, col_ind].sum() / n)
        
        return match_cost

    def robosac_map(self, num_agent, t_matrix, attack_type, attack_conf, data_dict, dataset, pred_gt_box_tensor, ego_output, num, batch_dict, gt_box_tensor, target_box, target_box_tensor, target_score, label, robosac_path, save_dir, attack):
        
        # Given Step Budget N and Sampling Set Size s, perform predictions
        
        ego_idx = 0
        all_agent_list = [i for i in range(num_agent)]
        all_agent_list.remove(ego_idx)
        collab_agent_list = []
        
        # import pdb; pdb.set_trace()
        if self.robosac_k == None:
            consensus_set_size = self.cal_robosac_consensus(num_agent, self.step_budget, self.number_of_attackers)

            if(consensus_set_size < 1):
                print('Expected Consensus Agent below 1. Exit.'.format(consensus_set_size))
                sys.exit()
        else:
            consensus_set_size = self.robosac_k
            
        # define attack model
        record_len = torch.tensor([consensus_set_size + 1]).cuda()
        self.attack_model = PGD(self.Model.cls_head, self.Model.att_fuse_module, self.Model.reg_head, record_len, t_matrix, self.Model.backbone, self.Model.generate_predicted_boxes)
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
        
        found = False
        # NOTE: 0~step_budget-1
        for step in range(1, self.step_budget + 1):

            collab_agent_list = random.sample(all_agent_list, k=consensus_set_size)
            
            att_output_dict, pred_tensor, pred_score, pred_box = self.run_attack(data_dict, dataset, attack_type, attack_target, attack_conf, pred_gt_box_tensor, ego_output, num, batch_dict, actual_attack, attack_src, gt_box_tensor, collab_agent_list)

            # We use jaccard index to define the difference between two bbox sets
            jac_index = self.get_jaccard_index(pred_boxes=pred_box, target_boxes=target_box)
            # print("collab_idx_list: ", collab_agent_list, "jac_index: ", jac_index[0])
            if jac_index[0] < self.match_thresh:
                continue
            else:
                sus_agent_list = [i for i in all_agent_list if i not in collab_agent_list]
                found = True
                break
            
        if not found:
            # print('No consensus!')
            # Can't achieve consensus, so fall back to original ego only result
            final_output = ego_output
            final_box = target_box
            final_box_tensor = target_box_tensor
            final_score = target_score
            pred = 1
        else:
            final_output = att_output_dict
            final_box = pred_box
            final_box_tensor = pred_tensor
            final_score = pred_score
            pred = 0
        
        robosac_loss_list = [[jac_index, label]]
        import os
        if not os.path.exists(robosac_path + save_dir):
            os.makedirs(robosac_path + save_dir)
        np.save(robosac_path + save_dir + f'/sample_{num}.npy', robosac_loss_list)
        
        history = {}
        # update reference frame for next iteration
        history['output'] = final_output
        history['box'] = final_box
            
        return final_box_tensor, final_score, gt_box_tensor, [pred, label], history
    
    def probing(self, num_agent, t_matrix, attack_type, attack_conf, data_dict, dataset, pred_gt_box_tensor, ego_output, num, batch_dict, gt_box_tensor, target_box, target_box_tensor, target_score, label, robosac_path, save_dir, attack):
        
        ego_idx = 0
        all_agent_list = [i for i in range(num_agent)]
        all_agent_list.remove(ego_idx)
        collab_agent_list = []
        
        # probing_step_limit_by_attacker_ratio
        NMax = []
        NTry = [0] * len(self.estimate_att_ratio)
        total_sampling_step = 0
        tmp_num_agent = int(num_agent)
        
        for ratio in self.estimate_att_ratio:
            
            temp_num_attackers = round((tmp_num_agent - 1) * (ratio))
            temp_num_consensus = (tmp_num_agent - 1) - temp_num_attackers
            NMax.append(self.cal_robosac_steps(tmp_num_agent, temp_num_consensus, temp_num_attackers))
        found = False
        NMax[0] = 1
            
        step = 0
        # ensuring probing tries will traverse all possible attacker ratios
        assert self.step_budget >= len(self.estimate_att_ratio)
        
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
        
        while step < self.step_budget and NTry < NMax:
            
            for i in range(len(self.estimate_att_ratio)):
                temp_attacker_ratio = self.estimate_att_ratio[i]
                consensus_set_size = round((tmp_num_agent - 1) * (1-temp_attacker_ratio))
                if NTry[i] < NMax[i]:
                    # print("Probing {} agents for consensus".format(consensus_set_size))
                    step += 1
                    total_sampling_step += 1
                    
                    # define attack model
                    record_len = torch.tensor([consensus_set_size + 1]).cuda()
                    self.attack_model = PGD(self.Model.cls_head, self.Model.att_fuse_module, self.Model.reg_head, record_len, t_matrix, self.Model.backbone, self.Model.generate_predicted_boxes)
                    
                    # probing_step_tried_by_consensus_set_size[consensus_set_size] += 1
                    # step budget available for probing
                    # try to probe attacker ratio
                    collab_agent_list = random.sample(
                    all_agent_list, k=consensus_set_size)
                    att_output_dict, pred_tensor, pred_score, pred_box = self.run_attack(data_dict, dataset, attack_type, attack_target, attack_conf, pred_gt_box_tensor, ego_output, num, batch_dict, actual_attack, attack_src, gt_box_tensor, collab_agent_list)
                    
                    jac_index = self.get_jaccard_index(pred_boxes=pred_box, target_boxes=target_box)

                    if jac_index[0] < self.match_thresh:

                        NTry[i] += 1 
                        continue
                    else:
                        # succeed to reach consensus
                        sus_agent_list = [
                            i for i in all_agent_list if i not in collab_agent_list]
                        found = True
                        break
                
        if not found:
            # print('No consensus!')
            # Can't achieve consensus, so fall back to original ego only result
            final_output = ego_output
            final_box = target_box
            final_box_tensor = target_box_tensor
            final_score = target_score
            pred = 1
        else:
            
            final_output = att_output_dict
            final_box = pred_box
            final_box_tensor = pred_tensor
            final_score = pred_score
            pred = 0
        
        robosac_loss_list = [[jac_index, label]]
        import os
        if not os.path.exists(robosac_path + save_dir):
            os.makedirs(robosac_path + save_dir)
        np.save(robosac_path + save_dir + f'/sample_{num}.npy', robosac_loss_list)
        
        history = {}
        # update reference frame for next iteration
        history['output'] = final_output
        history['box'] = final_box
            
        return final_box_tensor, final_score, gt_box_tensor, [pred, label], history
    
    def forward(self, data_dict, dataset, attack_conf, num = 0, attack = None, attack_type = 'pgd', save_dir = 'validation'):
        
        cav_content = data_dict['ego']
        num_agent = cav_content['record_len']
        output_dict = OrderedDict()
        if attack_conf != None:
            self.number_of_attackers = attack_conf.attack.n_att
        else:
            self.number_of_attackers = 1
        robosac_path = f'generate_robosac/{self.rob_type}/robosac_k_{self.robosac_k}_step_budget_{self.step_budget}_thresh_{self.match_thresh}/' 
        
        history_start = data_dict.pop('history')
        
        # ego-output
        ego_output, _ = self.Model.forward(data_dict=cav_content, if_single=True, dataset=dataset)
        output_dict['ego'] = ego_output
        with torch.no_grad():
            target_box_tensor, target_score, gt_box_tensor, target_bbox = \
                dataset.post_process(data_dict, output_dict)
        target_box = {'box_tensor':target_bbox, 'score':target_score}
        
        # 1. get reference result
        if (self.use_history and history_start == None) or not self.use_history:
            # ego-data generate
            target_output = ego_output
            target_box = {'box_tensor':target_bbox, 'score':target_score}
        else:
            target_output = history_start['output']
            target_box = history_start['box']
        
        if num_agent == 1 or attack == None:
            import os
            if not os.path.exists(robosac_path + save_dir):
                os.makedirs(robosac_path + save_dir)
            np.save(robosac_path + save_dir + f'/sample_{num}.npy',[])
            return target_box_tensor, target_score, gt_box_tensor, [], None
        
        # 2. generate batch dict
        batch_dict, t_matrix, record_len = self.Model.generate_batch_dict(cav_content)
        
        # 3. generate pred_gt_box_tensor
        no_att_output_dict, _ = self.Model.forward(data_dict=cav_content, dataset=dataset)
        output_dict = OrderedDict()
        output_dict['ego'] = no_att_output_dict
        with torch.no_grad():
            pred_gt_box_tensor, pred_score, gt_box_tensor, bbox = \
            dataset.post_process(data_dict, output_dict)
            
        if attack_type == 'no_attack':
            label = 0
        else:
            label = 1
        
        # different defense ways
        if self.rob_type == 'probing':
            
            return self.probing(num_agent, t_matrix, attack_type, attack_conf, data_dict, dataset, pred_gt_box_tensor, ego_output, num, batch_dict, gt_box_tensor, target_box, target_box_tensor, target_score, label, robosac_path, save_dir, attack)
            
        
        elif self.rob_type == 'robosac_mAP':
            
            final_box_tensor, final_score, gt_box_tensor, pred_label_list, history = self.robosac_map(num_agent, t_matrix, attack_type, attack_conf, data_dict, dataset, pred_gt_box_tensor, ego_output, num, batch_dict, gt_box_tensor, target_box, target_box_tensor, target_score, label, robosac_path, save_dir, attack)
                
            return final_box_tensor, final_score, gt_box_tensor, pred_label_list, history


# def main():
    
#     robosac = ROBOSAC()
#     robosac.forward(data_dict, dataset, attack_conf, num = 0, attack = None, attack_type = 'pgd')