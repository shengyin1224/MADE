import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import sys
import random
import math

from .utils import label_attacker, rm_com_pair

class ROBOSAC_Detector(nn.Module):

    def __init__(self, rob_type = 'robosac_mAP', robosac_k = None, step_budget = 3, match_thresh = 0.3, use_history = False, estimate_att_ratio = [0.0, 0.2, 0.4, 0.6, 0.8]):

        '''
        Paras:
        
            rob_type: robosac_mAP/adaptive/fix_attackers
            robosac_k: specify consensus set size if needed
            step_budget: sampling budget in a single frame
            match_thresh: IoU threshold for validating two detection results
            use_history: use history frame for computing the consensus, reduce 1 step of forward prop
            
        '''

        super().__init__()
        self.rob_type = rob_type
        self.robosac_k = robosac_k
        self.step_budget = step_budget
        self.match_thresh = match_thresh
        self.use_history = use_history
        self.estimate_att_ratio = estimate_att_ratio
    
    def cal_robosac_consensus(self, num_agent, step_budget, num_attackers):
        num_agent = int(num_agent - 1)
        step_budget = int(step_budget)
        num_attackers = int(num_attackers)
        eta = num_attackers / num_agent
        s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
        return s
    
    def cal_robosac_steps(self, num_agent, num_consensus, num_attackers):
        # exclude ego agent
        # import pdb; pdb.set_trace()
        num_agent = num_agent - 1
        eta = float(num_attackers / num_agent)
        N = np.ceil(np.log(1 - 0.99) / np.log(1 - np.power(1 - eta, int(num_consensus)))).astype(int)
        return N
    
    def random_sample_per_agent(self, other_fuse_idx_list, k = 1, agent_idx = 0):

        tmp_list = []
        for idx, item in enumerate(other_fuse_idx_list):

            if idx == agent_idx:
                tmp = random.sample(item, k=k)
                tmp.append(agent_idx)
                tmp_list.append(tmp)
            else:
                tmp_list.append(item)

        return tmp_list
    
    def select_element(self, collab_idx_list, number_of_agent):

        final_com_src, final_com_tgt = [], []
        for idx in range(number_of_agent):

            colla_list = collab_idx_list[idx]

            for agent in colla_list:
                final_com_src.append([0, agent])
                final_com_tgt.append([0, idx])
        
        final_com_src = torch.tensor(final_com_src).cuda()
        final_com_tgt = torch.tensor(final_com_tgt).cuda()

        return final_com_src, final_com_tgt

    def get_final_com_pair(self, final_collab_list, number_of_agent):

        final_src, final_tgt = [], []
        for agent_idx in range(number_of_agent):

            tmp_collab_list = final_collab_list[agent_idx]

            for agent in tmp_collab_list:
                final_src.append([0, agent])
                final_tgt.append([0, agent_idx])
        
        return torch.tensor(final_src).cuda(), torch.tensor(final_tgt).cuda()
        

    def robosac_map(self, num_agent, attack, attack_src, attack_tgt, model, anchors, bev_seq, trans_matrices, batch_size, target_result):

        number_of_agent = num_agent[0, 0]
        if attack_src == None:
            self.number_of_attackers = 0
        else:
            self.number_of_attackers = len(attack_src) / number_of_agent

        # compute consensus set size
        if self.robosac_k == None:
            consensus_set_size = self.cal_robosac_consensus(number_of_agent, self.step_budget, self.number_of_attackers)

            # if(consensus_set_size < 1):
            #     print('Expected Consensus Agent below 1. Exit.'.format(consensus_set_size))
            #     sys.exit()
        else:
            consensus_set_size = self.robosac_k

        final_collab_list = []
        for agent_idx in range(number_of_agent):

            found = False
            other_fuse_idx_list = []
            for i in range(number_of_agent):
                if i == agent_idx:
                    other_fuse_idx_list.append([j for j in range(number_of_agent) if j != i])
                else:
                    other_fuse_idx_list.append([])
            
            if self.number_of_attackers + 1 + consensus_set_size <= number_of_agent:

                for step in range(1, self.step_budget + 1):

                    collab_idx_list = self.random_sample_per_agent(other_fuse_idx_list, k = consensus_set_size, agent_idx = agent_idx)
                    tmp_com_src, tmp_com_tgt = self.select_element(collab_idx_list, number_of_agent)

                    att_result, att_box = model(bev_seq, trans_matrices, num_agent, batch_size=batch_size,
                    com_src=tmp_com_src, com_tgt=tmp_com_tgt, 
                    attack=attack, attack_src=attack_src, attack_tgt=attack_tgt,
                    batch_anchors=anchors, nms=True)

                    att_final_result = att_box
                    jac_index = model.get_jaccard_index(att_final_result, target_result)

                    # print("collab_idx_list: ", collab_idx_list, "jac_index: ", jac_index[0][agent_idx], "attack_src: ", attack_src, "attack_tgt: ", attack_tgt)
                    
                    if jac_index[0][agent_idx] < self.match_thresh:
                        continue
                    else:
                        found = True
                        break
                
            if not found:
                tmp_collb_list = [agent_idx]
            else:
                tmp_collb_list = collab_idx_list[agent_idx]
            final_collab_list.append(tmp_collb_list)

        final_com_src, final_com_tgt = self.get_final_com_pair(final_collab_list, number_of_agent)

        return final_com_src, final_com_tgt


    def probing(self, num_agent, attack, attack_src, attack_tgt, model, anchors, bev_seq, trans_matrices, batch_size, target_result):

        number_of_agent = num_agent[0, 0]
        estimated_attacker_ratio = 1.0

        final_collab_list = []
        NTry = [0] * len(self.estimate_att_ratio)

        # probing_step_limit_by_attacker_ratio
        NMax = []
        
        total_sampling_step = 0
        tmp_num_agent = int(number_of_agent)

        for ratio in self.estimate_att_ratio:
            temp_num_attackers = round((tmp_num_agent - 1) * (ratio))
            temp_num_consensus = (tmp_num_agent - 1) - temp_num_attackers
            NMax.append(self.cal_robosac_steps(tmp_num_agent, temp_num_consensus, temp_num_attackers))
        NMax[0] = 1

        for agent_idx in range(number_of_agent):

            found = False
            other_fuse_idx_list = []
            for i in range(number_of_agent):
                if i == agent_idx:
                    other_fuse_idx_list.append([j for j in range(number_of_agent) if j != i])
                else:
                    other_fuse_idx_list.append([])
                
            step = 0
            # if number_of_agent == 4:
            #     import pdb; pdb.set_trace()
            assert self.step_budget >= len(self.estimate_att_ratio)

            while step < self.step_budget and NTry < NMax:
            
                for i in range(len(self.estimate_att_ratio)):
                    temp_attacker_ratio = self.estimate_att_ratio[i]
                    consensus_set_size = round((tmp_num_agent - 1) * (1-temp_attacker_ratio))
                    if NTry[i] < NMax[i]:

                        step += 1
                        total_sampling_step += 1
                        collab_idx_list = self.random_sample_per_agent(other_fuse_idx_list, k = consensus_set_size, agent_idx = agent_idx)
                        tmp_com_src, tmp_com_tgt = self.select_element(collab_idx_list, number_of_agent)

                        att_result, att_box = model(bev_seq, trans_matrices, num_agent, batch_size=batch_size,
                        com_src=tmp_com_src, com_tgt=tmp_com_tgt, 
                        attack=attack, attack_src=attack_src, attack_tgt=attack_tgt,
                        batch_anchors=anchors, nms=True)

                        att_final_result = att_box
                        jac_index = model.get_jaccard_index(att_final_result, target_result)

                        # print("i: ", i, "collab_idx_list: ", collab_idx_list, "jac_index: ", jac_index[0][agent_idx], "temp_attacker_ratio: ",temp_attacker_ratio, "step: ", step, "NTry: ", NTry, "estimated_ratio: ", estimated_attacker_ratio)
                        # print("tmp_com_src: ", tmp_com_src, "tmp_com_tgt: ", tmp_com_tgt)

                        if jac_index[0][agent_idx] < self.match_thresh:
                            NTry[i] += 1
                            continue 
                        else:
                            found = True
                            infer_collb_list = collab_idx_list[agent_idx]
                            if temp_attacker_ratio < estimated_attacker_ratio:
                                estimated_attacker_ratio = temp_attacker_ratio
   
                                for j in range(i, len(self.estimate_att_ratio)):
                                    # set all the larger attacker ratio to 0
                                    NTry[j] = NMax[j]
                                break
                
            if not found:
                tmp_collb_list = [agent_idx]
            else:
                tmp_collb_list = infer_collb_list
            final_collab_list.append(tmp_collb_list)

        final_com_src, final_com_tgt = self.get_final_com_pair(final_collab_list, number_of_agent)

        return final_com_src, final_com_tgt

    def forward(self, 
                model: torch.nn.Module,
                bev: torch.Tensor,
                trans_matrices: torch.Tensor,
                num_agent: torch.Tensor,
                anchors: torch.Tensor,
                attack: torch.Tensor,
                attack_src: torch.Tensor,
                attack_tgt: torch.Tensor,
                batch_size = 1,
                history_result = None):
        
        # generate srcs, tgts, att_srcs
        com_srcs, com_tgts = model.get_attack_det_com_pairs(num_agent)

        # generate ego result
        results_list = model.multi_com_forward(
            bev, trans_matrices, com_srcs, com_tgts, attack, attack_src, attack_tgt, batch_size)
        k = num_agent[0, 0]
        box_list = [model.post_process(results, anchors, k) for results in results_list]
        # num_agent elements [0-0, 1-1, 2-2, 3-3]
        ego_result = box_list[0]

        # get reference result
        if (self.use_history and history_result == None) or not self.use_history:
            # ego-data generate
            target_result = ego_result
        else:
            target_result = history_result
        
        # different defense way
        if self.rob_type == 'robosac_mAP':

            com_src, com_tgt = self.robosac_map(num_agent, attack, attack_src, attack_tgt, model, anchors, bev, trans_matrices, batch_size, target_result)

        elif self.rob_type == 'probing':

            com_src, com_tgt = self.probing(num_agent, attack, attack_src, attack_tgt, model, anchors, bev, trans_matrices, batch_size, target_result)


        return com_src, com_tgt, ego_result