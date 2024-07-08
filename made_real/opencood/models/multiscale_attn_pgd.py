# modified from from torchattacks
from asyncio import FastChildWatcher
from codecs import decode
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops import diff_iou_rotated_3d
from collections import OrderedDict
from torchattacks.attack import Attack
from opencood.utils import eval_utils
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
import matplotlib.pyplot as plt
from opencood.utils.box_utils import corner_to_center, corner_to_center_torch
import time

import os 
import sys 
import ipdb

def draw_gradient_map(gradient,num):
    
    eps = "[5,0,0]"
    attack = 'pgd'
    gradient = torch.abs(gradient).squeeze(0)
    tmp_tensor = gradient.mean(dim=0)
    # tmp_tensor = (tmp_tensor - min_x) / (max_x - min_x)
    tmp_tensor = F.normalize(tmp_tensor)

    plt.imshow(tmp_tensor.detach().cpu().numpy(), cmap=plt.cm.hot)
    plt.colorbar()

    plt.tight_layout()
    plt.title(f'sample_{num}')
    save_path = '/GPFS/data/shengyin/OpencoodV2-Main/OpenCOODv2/outcome/gradient_visualize/'
    
    save_path = save_path + f'eps{eps}_{attack}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(save_path + f'sample_{num}.png')
    plt.close()


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, cls_head, fuse_model, reg_head, record_len, t_matrix, backbone, generate_predicted_boxes,
                eps=0.1, alpha=0.1, steps=40, 
                attack_mode='others',
                n_att=1, colla_attack=True, 
                noise_attack=False,
                random_start=True,
                project=True,
                save_path=None,
                save_attack=False):
        super().__init__("PGD", cls_head)
        """
            需要支持的功能：
            1. 设置攻击的超参数(eps, alpha, steps, project) -- done
            2. 设置attacker的src和tgt -- done
            3. 设置attacker个数 -- to do
            4. targeted 攻击？ -- to do
            5. agent 之间的迁移 -- done 
            6. random smooth -- to do
            7. multi-agent attack 是否合作 -- to do
        """
        self.eps = eps
        self.alpha = alpha  
        self.gamma = 1
        self.steps = steps
        self.random_start = random_start
        self.attack_mode = attack_mode
        self.noise_attack = noise_attack
        self.project = project
        self.record_len = record_len
        self.t_matrix = t_matrix
        self.fuse_model = fuse_model
        self.model = cls_head

        self.backbone = backbone
        self.cls_head = cls_head
        self.reg_head = reg_head
        self.generate_predicted_boxes = generate_predicted_boxes

        self.n_att = n_att
        self.colla_attack = colla_attack

    def model_run(self, data_dict, num = 0, attack_target = 'pred', 
                  shift_feature = False, rotate_feature = False, attack_conf = None, 
                  real_data_dict = None, attack_srcs = [], if_erase = False, erase_index = [], 
                  dataset = None, pred_gt_box_tensor = None, shift_dir_of_box = [], attack = None, 
                  if_fuse = True, if_inference = False, gt_box_tensor = None, cls = None,
                  if_att_score = False, if_shift_attack = False, shift_attack = None):

        """
        由于where2comm下的结构比较复杂,所以这里单独构造一个函数跑函数结果
        """

        feature_list = data_dict['feature_list']

        if if_shift_attack:
            shift_attack = self.fuse_model(feature_list,
                                            self.record_len,
                                            self.t_matrix, 
                                            data_dict = data_dict, attack = attack, 
                                            attack_src = attack_srcs, num = num, 
                                            shift_feature = shift_feature, rotate_feature = rotate_feature,
                                            attack_conf = attack_conf, real_data_dict = real_data_dict, if_erase = if_erase, 
                                            erase_index = erase_index, attack_target = attack_target, 
                                            pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, 
                                            shift_dir_of_box = shift_dir_of_box, if_fuse = if_fuse, if_inference = if_inference, gt_box_tensor = gt_box_tensor,
                                            cls = cls, if_att_score = if_att_score, if_shift_attack = if_shift_attack)
            return shift_attack

        # if self.multi_scale:
        if if_att_score:
            fused_feature, _, attention_score = self.fuse_model(feature_list,
                                            self.record_len,
                                            self.t_matrix, 
                                            data_dict = data_dict, attack = attack, 
                                            attack_src = attack_srcs, num = num, 
                                            shift_feature = shift_feature, rotate_feature = rotate_feature,
                                            attack_conf = attack_conf, real_data_dict = real_data_dict, if_erase = if_erase, 
                                            erase_index = erase_index, attack_target = attack_target, 
                                            pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, 
                                            shift_dir_of_box = shift_dir_of_box, if_fuse = if_fuse, if_inference = if_inference, gt_box_tensor = gt_box_tensor,
                                            cls = cls, if_att_score = if_att_score, shift_attack = shift_attack)
        else:
            fused_feature, _ = self.fuse_model(feature_list,
                                            self.record_len,
                                            self.t_matrix, 
                                            data_dict = data_dict, attack = attack, 
                                            attack_src = attack_srcs, num = num, 
                                            shift_feature = shift_feature, rotate_feature = rotate_feature,
                                            attack_conf = attack_conf, real_data_dict = real_data_dict, if_erase = if_erase, 
                                            erase_index = erase_index, attack_target = attack_target, 
                                            pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, 
                                            shift_dir_of_box = shift_dir_of_box, if_fuse = if_fuse, if_inference = if_inference, gt_box_tensor = gt_box_tensor,
                                            cls = cls, if_att_score = if_att_score, shift_attack = shift_attack)
        
        if if_att_score:
            return fused_feature, _, attention_score
        else:
            return fused_feature, _


    def forward(self, data_dict, anchors, reg_targets, labels, num = 0, sparsity = False, 
        keep_pos = False, attack_target = 'pred', shift_feature = False, rotate_feature = False, 
        attack_conf = None, real_data_dict = None, attack_srcs = [], if_erase = False, 
        erase_index = [], dataset = None, pred_gt_box_tensor = None, shift_dir_of_box = [], 
        gt_box_tensor = None, att_layer = []):
        r"""
        Overridden.
        data_dict: actually batch_dict
        """

        # # feature
        # 4个参与loss计算的变量
        anchors = anchors.clone().detach().to(self.device)
        reg_targets = reg_targets.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)


        # 如果攻击者数大于可传递信息的智能体数，直接返回空的attack
        # import ipdb; ipdb.set_trace()
        if self.record_len - 1 < self.n_att or pred_gt_box_tensor == None:
            return [torch.zeros(64, 100, 252).cuda(), torch.zeros(128, 50, 126).cuda(), torch.zeros(256, 25, 63).cuda()], attack_srcs

        # 计算抹除不成功的index
        if if_erase:

            tmp_erase_index = erase_index
            if attack_target == 'gt':
                object_bbx_center = real_data_dict['object_bbx_center']
                object_bbx_mask = real_data_dict['object_bbx_mask']
                object_bbx_center = object_bbx_center[object_bbx_mask == 1]
            else:
                object_bbx_center = corner_to_center_torch(corner3d=pred_gt_box_tensor, order='hwl')

            shift_index = []
            for j in range(object_bbx_center.shape[0]):
                if j not in tmp_erase_index:
                    shift_index.append(j)

            tmp_erase_center = object_bbx_center[tmp_erase_index]
            tmp_shift_center = object_bbx_center[shift_index]
            shift_center = torch.zeros(size=(100,7))
            erase_center = torch.zeros(size=(100,7))
            shift_center[:len(shift_index)], erase_center[:len(tmp_erase_index)] = tmp_shift_center, tmp_erase_center
            
            shift_mask = torch.zeros(size=(100,))
            erase_mask = torch.zeros(size=(100,))
            shift_mask[:len(shift_index)], erase_mask[:len(tmp_erase_index)] = 1, 1

            tmp_anchors = anchors.clone()
            tmp_anchors = tmp_anchors.cpu().detach().numpy()
            shift_mask = shift_mask.numpy()
            erase_mask = erase_mask.numpy()
            shift_center = shift_center.numpy()
            erase_center = erase_center.numpy()
            
            label_shift = dataset.post_processor.generate_label(gt_box_center=shift_center,
                anchors=tmp_anchors,
                mask=shift_mask)
            label_erase = dataset.post_processor.generate_label(gt_box_center=erase_center,
                anchors=tmp_anchors,
                mask=erase_mask)
            
            label_shift = torch.tensor(label_shift['pos_equal_one']).cuda()
            label_erase = torch.tensor(label_erase['pos_equal_one']).cuda()


        # print(f"time is {end - start}.")
        # 是否联合攻击
        if self.colla_attack:
            # 让attacks变成拥有 3 * n_att个元素
            attack = [torch.Tensor(self.n_att, 64, 100, 252).cuda(), torch.Tensor(self.n_att, 128, 50, 126).cuda(), torch.Tensor(self.n_att, 256, 25, 63).cuda()]

            if self.random_start:
                    # Starting at a uniformly random point
                    if not isinstance(self.eps, float):
                        for j in range(3):
                            attack[j].uniform_(-self.eps[j], self.eps[j])
                    else:
                        for a in attack:
                            a.uniform_(-self.eps, self.eps)

            if self.noise_attack: 
                    for _ in range(self.steps):
                        for a in attack:
                            a += self.alpha * torch.randn_like(a) * 3
                        if self.project:
                            if not isinstance(self.eps, float):
                                for j in range(3):
                                    attack[j].clamp_(min=-self.eps[j], max=self.eps[j])
                            else:
                                for a in attack:
                                    a.clamp_(min=-self.eps, max=self.eps)
            else:
                loss_list = []
                for step_m in range(self.steps):
                    
                    # require grad
                    for a in attack:
                        a.requires_grad = True
                    
                    # 输入inner model
                    outputs, _ = self.model_run(data_dict, attack = attack, attack_srcs = attack_srcs, shift_feature = shift_feature, rotate_feature = rotate_feature, attack_conf = attack_conf, real_data_dict = real_data_dict, if_erase = if_erase, erase_index = erase_index, num = num, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box, gt_box_tensor = gt_box_tensor)

                    cost = self.loss(outputs, anchors, reg_targets, labels, attack_target, tmp_erase_index, step_m, num)

                    # cost = self.new_loss(outputs, anchors, reg_targets, labels, label_shift, label_erase, attack_target, tmp_erase_index)

                    # 存储cost测试一下
                    tmp = torch.clone(cost).detach().cpu().numpy()
                    loss_list.append(tmp)

                    # Update adversarial images
                    grad_list = torch.autograd.grad(cost, attack,
                                                retain_graph=False, create_graph=False)
                    grad_list = list(grad_list)

                    # FGSN的计算公式
                    for k in range(len(attack)):
                        a = attack[k]
                        a = a.detach() - self.alpha[k] * grad[k].sign()
                        a[grad[k] == 0] = 0
                        attack[k] = a
                    if self.project:
                        if not isinstance(self.eps, float):
                            for j in range(3):
                                attack[j].clamp_(min=-self.eps[j], max=self.eps[j])
                        else:
                                for a in attack:
                                    a.clamp_(min=-self.eps, max=self.eps)

            np.save(f'/GPFS/data/shengyin/OpenCOOD-main/attack_loss_1/loss_sample_{num}.npy',loss_list)
            attacks = attack
        else:
            # if num == 21:
            #     print(14)
            attacks = []
            for attack_src in attack_srcs:

                attack = [torch.Tensor(1, 64, 100, 252).cuda(), torch.Tensor(1, 128, 50, 126).cuda(), torch.Tensor(1, 256, 25, 63).cuda()]

                # start = time.time()

                if self.random_start:
                    # Starting at a uniformly random point
                    if not isinstance(self.eps, float):
                        for i in range(len(attack)):
                            # attack[i].uniform_(-self.eps[i], self.eps[i])
                            attack[i].uniform_(-self.eps[i], self.eps[i])
                    else:
                        for a in attack:
                            a.uniform_(-self.eps[i], self.eps[i])

                if self.noise_attack: 
                    for _ in range(self.steps):
                        for ki in range(len(attack)):
                            a = attack[ki]
                            a += self.alpha[ki] * torch.randn_like(a) * 3
                        if self.project:
                            if not isinstance(self.eps, float):
                                for i in range(len(attack)):
                                    attack[i].clamp_(min=-self.eps[i], max=self.eps[i])
                            else:
                                for a in attack:
                                    a.clamp_(min=-self.eps, max=self.eps)
                else:
                    loss_list = []

                    # 生成erase/shift的attack，后面直接用
                    if shift_feature:
                        shift_attack = self.model_run(data_dict, attack = attack, 
                            attack_srcs = [attack_src], shift_feature = shift_feature, 
                            rotate_feature = rotate_feature, attack_conf = attack_conf, real_data_dict = real_data_dict, 
                            if_erase = if_erase, erase_index = erase_index, num = num, attack_target = attack_target, 
                            pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box,
                            gt_box_tensor = gt_box_tensor, if_shift_attack = True, if_att_score = True)
                    else:
                        shift_attack = None

                    # import ipdb; ipdb.set_trace()
                    for p in range(self.steps):
                        
                        # require grad
                        for a in attack:
                            a.requires_grad = True
                        
                        # 输入inner model
                        # TODO: 目前attention_score仅为第一层的，shape为(100, 252)
                        outputs, _, attention_score = self.model_run(data_dict, attack = attack, 
                            attack_srcs = [attack_src], shift_feature = shift_feature, 
                            rotate_feature = rotate_feature, attack_conf = attack_conf, real_data_dict = real_data_dict, 
                            if_erase = if_erase, erase_index = erase_index, num = num, attack_target = attack_target, 
                            pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box,
                            gt_box_tensor = gt_box_tensor, if_att_score = True, shift_attack = shift_attack)

                        # if if_erase:
                        #     cost, true_num = self.new_loss(outputs, anchors, reg_targets, labels, label_shift, label_erase, attack_target, tmp_erase_index, shift_index)
                        # else:
                        cost, true_num = self.loss(outputs, anchors, reg_targets, labels, attack_target, erase_index, p, num)

                        # cost, true_num, attention_score = self.attention_score_loss(outputs, anchors, reg_targets, labels, label_shift, label_erase, attack_target, 
                        #                 tmp_erase_index, shift_index, attention_score)

                        # att_score_path = 'outcome/attention_score/save' + f'/sample_{num}'
                        # if not os.path.exists(att_score_path):
                        #     os.makedirs(att_score_path)
                        # np.save(att_score_path + f'/iter_{p}.npy', attention_score.clone().detach().cpu().numpy())

                        if cost.isnan():
                            attack = [torch.zeros(64, 100, 252).cuda(), torch.zeros(128, 50, 126).cuda(), torch.zeros(256, 25, 63).cuda()]
                            break    

                        # Update adversarial images

                        # grad = torch.autograd.grad(cost, attack,
                        #                          retain_graph=True, create_graph=False, allow_unused=True)
                        
                        grad = torch.autograd.grad(cost, attack,
                                                retain_graph=False, create_graph=False)
                        grad = list(grad)

                        # FGSN的计算公式

                        tmp_loss = torch.clone(cost)
                        loss_list.append(tmp_loss.detach().cpu().numpy())

                        for k in range(len(attack)):
                            a = attack[k]
                            if grad[k] == None:
                                grad[k] = torch.zeros(a.shape).cuda()
                        
                            # import ipdb; ipdb.set_trace()
                            a = a.detach() - self.alpha[k] * grad[k].sign()
                            a[grad[k] == 0] = 0
                            # a = a.detach() - self.alpha * grad[k] # 梯度下降
                            attack[k] = a
                        if self.project:
                            if not isinstance(self.eps, float):
                                for i in range(len(attack)):
                                    attack[i].clamp_(min=-self.eps[i], max=self.eps[i])
                            else:
                                for a in attack:
                                    a.clamp_(min=-self.eps, max=self.eps)
                        

                    # loss_save_path = 'outcome/loss_list' + f'/PGD_eps[1.5,1.5,0]_step{self.steps}'
                    # if os.path.exists(loss_save_path) == False:
                    #     os.makedirs(loss_save_path)
                    # np.save(loss_save_path + f'/loss_sample_{num}.npy',loss_list)

                attacks.extend(attack)
            # attacks = torch.cat(attacks, dim=0)
            real_attacks = [torch.Tensor(self.n_att, 64, 100, 252).cuda(), torch.Tensor(self.n_att, 128, 50, 126).cuda(), torch.Tensor(self.n_att, 256, 25, 63).cuda()]
            for block in range(3):
                for att in range(self.n_att):
                    real_attacks[block][att] = attacks[3*att + block]
            attacks = real_attacks
        return attacks, attack_srcs

    def get_attack_src(self, agent_num):

        if agent_num - 1 < self.n_att:
            return []

        attacker = torch.randint(low=1,high=agent_num,size=(self.n_att,))
        tmp = []
        for i in range(len(attacker)):
            tmp.append(attacker[i])

        return tmp
    
    def inference(self, data_dict, attack, attack_src, delete_list = [], num = 0, shift_feature = False, rotate_feature = False, attack_conf = None, real_data_dict = None, if_erase = False, erase_index = [], attack_target = 'pred', pred_gt_box_tensor = None, dataset = None, shift_dir_of_box = [], if_inference = True, gt_box_tensor = None, cls = None, collab_agent_list = None): 

        if delete_list != []:
            if_fuse = False
        else:
            if_fuse = True
        
        if collab_agent_list != None:
            
            if collab_agent_list == []:
                if_fuse = False
            else:
                if_fuse = True
              
        outputs, residual_vector = self.model_run(data_dict, attack = attack, attack_srcs = attack_src, num = num, shift_feature = shift_feature, rotate_feature = rotate_feature, attack_conf = attack_conf, real_data_dict = real_data_dict, if_erase = if_erase, erase_index = erase_index, attack_target = attack_target, pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, shift_dir_of_box = shift_dir_of_box, if_fuse = if_fuse, if_inference = if_inference, gt_box_tensor = gt_box_tensor, cls = cls)
        return outputs, residual_vector


    def attention_score_loss(self,
            result, # backbone的输出
            anchors, # (1, H, W, 2, 7)
            reg_targets, # (B, H, W, 14)
            labels,      # (B, H, W, 2)  
            label_shift, # (B, H, W, 2)
            label_erase, # (B, H, W, 2)
            attack_target, # pred / gt 
            erase_index,
            shift_index,
            attention_score, # (H, W)
            ):
        
        spatial_features_2d = result
        pred_cls = self.cls_head(spatial_features_2d) # (B, 2, H, W)
        pred_loc = self.reg_head(spatial_features_2d) # (B, 14, H, W)

        _, bbox_temp = self.generate_predicted_boxes(pred_cls, pred_loc)
        pred_loc = bbox_temp.reshape(1, 100, 252, 7).contiguous()

        # 调整shape
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().squeeze(0)                # (H, W, 2)
        pred_cls = pred_cls.view(-1, )  
        pred_cls = torch.sigmoid(pred_cls)                                             # (2*H*W, 1)
        anchors = anchors.squeeze(0)                                                  # (H, W, 2, 7)
        attention_score = attention_score.view(-1,) # (H * W, )

        # 先生成真实的pred，再选出foreground
        '''
        delta_to_boxes3d
        inputs: rm (B, H, W, 14), anchor (1, H, W, 2, 7)
        outputs: (B, H*W*2, 7)
        '''
        # decoded_pred = VoxelPostprocessor.delta_to_boxes3d(pred_loc, anchors).view(-1, 7)
        if attack_target == 'gt':
            decoded_target = VoxelPostprocessor.delta_to_boxes3d(reg_targets, anchors).view(-1, 7)
        else:
            decoded_target = reg_targets.view(-1,7)
        decoded_pred = pred_loc.view(-1,7)

        # gt/pred不同之处在于label的处理
        if attack_target == 'gt':
            label_shift = label_shift.squeeze(0).view(-1)    # (H*W, )
            fg_proposal_shift = label_shift == 1              # (H*W, )
            bg_proposal_shift = label_shift == 0              # (H*W, )

            label_erase = label_erase.squeeze(0).view(-1)    # (H*W, )
            fg_proposal_erase = label_erase == 1              # (H*W, )
            bg_proposal_erase = label_erase == 0              # (H*W, )

            labels = labels.squeeze(0).view(-1)    # (H*W, )
            fg_proposal = labels == 1              # (H*W, )
            bg_proposal = labels == 0              # (H*W, )
        else:
            label_shift = label_shift.squeeze(0).view(-1, )
            label_shift = torch.sigmoid(label_shift)
            fg_proposal_shift = label_shift > 0.5             # (H*W, )
            bg_proposal_shift = label_shift <= 0.5           # (H*W, )  

            label_erase = label_erase.squeeze(0).view(-1, )
            label_erase = torch.sigmoid(label_erase)
            fg_proposal_erase = label_erase > 0.5             # (H*W, )
            bg_proposal_erase = label_erase <= 0.5           # (H*W, ) 

            labels = labels.squeeze(0).view(-1, )
            labels = torch.sigmoid(labels)
            fg_proposal = labels > 0.5             # (H*W, )
            bg_proposal = labels <= 0.5           # (H*W, )  
    
        pred = decoded_pred[fg_proposal_shift].unsqueeze(0)       # (1, N, 7)
        target = decoded_target[fg_proposal_shift].unsqueeze(0)   # (1, N, 7)
        shift_attention_score = attention_score[fg_proposal_shift]

        # compute IoU
        # pred中7个特性的顺序是 (x,y,z,h,w,l,alpha)
        pred[:,:,[3,4]] = pred[:,:,[4,3]]
        target[:,:,[3,4]] = target[:,:,[4,3]]
        # input (1, N, 7), 其中7个特性为(x,y,z,w,h,l,alpha) 
        # output (1, N)
        if fg_proposal_shift.sum() == 0:
            iou = 0
        else:
            iou = diff_iou_rotated_3d(pred, target.float())[0].squeeze(0)
            iou = torch.clamp(iou, min=0, max=1)

        # loss 
        lamb = 0.00005
        att_lamb = 100
        # shift_fg_loss = 0
        shift_fg_loss = torch.sum(- torch.log(1 - pred_cls[fg_proposal_shift] + 1e-6) * iou)
        # erase_fg_loss = torch.sum(- torch.log(1 - pred_cls[fg_proposal_erase] + 1e-6))
        erase_fg_loss = 0
        fg_loss = shift_fg_loss + erase_fg_loss
        # bg_loss = torch.sum(- (1-pred_cls[bg_proposal]).pow(self.gamma) * torch.log(pred_cls[bg_proposal] + 1e-6))
        bg_loss = torch.zeros([1]).to(fg_loss.device)
        att_loss = torch.sum(shift_attention_score)
        total_loss =  (- att_lamb * att_loss).unsqueeze(0)
        total_loss += fg_loss + lamb * bg_loss

        return total_loss, fg_proposal.sum(), att_loss


    def new_loss(self,
            result, # backbone的输出
            anchors, # (1, H, W, 2, 7)
            reg_targets, # (B, H, W, 14)
            labels,      # (B, H, W, 2)  
            label_shift, # (B, H, W, 2)
            label_erase, # (B, H, W, 2)
            attack_target, # pred / gt 
            erase_index,
            shift_index
            ):
        
        spatial_features_2d = result
        pred_cls = self.cls_head(spatial_features_2d) # (B, 2, H, W)
        pred_loc = self.reg_head(spatial_features_2d) # (B, 14, H, W)

        _, bbox_temp = self.generate_predicted_boxes(pred_cls, pred_loc)
        pred_loc = bbox_temp.reshape(1, 100, 252, 7).contiguous()

        # 调整shape
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().squeeze(0)                # (H, W, 2)
        pred_cls = pred_cls.view(-1, )  
        pred_cls = torch.sigmoid(pred_cls)                                             # (2*H*W, 1)
        anchors = anchors.squeeze(0)                                                   # (H, W, 2, 7)
        
        # 先生成真实的pred，再选出foreground
        '''
        delta_to_boxes3d
        inputs: rm (B, H, W, 14), anchor (1, H, W, 2, 7)
        outputs: (B, H*W*2, 7)
        '''
        # decoded_pred = VoxelPostprocessor.delta_to_boxes3d(pred_loc, anchors).view(-1, 7)
        if attack_target == 'gt':
            decoded_target = VoxelPostprocessor.delta_to_boxes3d(reg_targets, anchors).view(-1, 7)
        else:
            decoded_target = reg_targets.view(-1,7)
        decoded_pred = pred_loc.view(-1,7)

        # gt/pred不同之处在于label的处理
        if attack_target == 'gt':
            label_shift = label_shift.squeeze(0).view(-1)    # (H*W, )
            fg_proposal_shift = label_shift == 1              # (H*W, )
            bg_proposal_shift = label_shift == 0              # (H*W, )

            label_erase = label_erase.squeeze(0).view(-1)    # (H*W, )
            fg_proposal_erase = label_erase == 1              # (H*W, )
            bg_proposal_erase = label_erase == 0              # (H*W, )

            labels = labels.squeeze(0).view(-1)    # (H*W, )
            fg_proposal = labels == 1              # (H*W, )
            bg_proposal = labels == 0              # (H*W, )
        else:
            label_shift = label_shift.squeeze(0).view(-1, )
            label_shift = torch.sigmoid(label_shift)
            fg_proposal_shift = label_shift > 0.5             # (H*W, )
            bg_proposal_shift = label_shift <= 0.5           # (H*W, )  

            label_erase = label_erase.squeeze(0).view(-1, )
            label_erase = torch.sigmoid(label_erase)
            fg_proposal_erase = label_erase > 0.5             # (H*W, )
            bg_proposal_erase = label_erase <= 0.5           # (H*W, ) 

            labels = labels.squeeze(0).view(-1, )
            labels = torch.sigmoid(labels)
            fg_proposal = labels > 0.5             # (H*W, )
            bg_proposal = labels <= 0.5           # (H*W, )  
    
        pred = decoded_pred[fg_proposal_shift].unsqueeze(0)       # (1, N, 7)
        target = decoded_target[fg_proposal_shift].unsqueeze(0)   # (1, N, 7)

        # compute IoU
        # pred中7个特性的顺序是 (x,y,z,h,w,l,alpha)
        pred[:,:,[3,4]] = pred[:,:,[4,3]]
        target[:,:,[3,4]] = target[:,:,[4,3]]
        # input (1, N, 7), 其中7个特性为(x,y,z,w,h,l,alpha) 
        # output (1, N)
        if fg_proposal_shift.sum() == 0:
            iou = 0
        else:
            iou = diff_iou_rotated_3d(pred, target.float())[0].squeeze(0)
            iou = torch.clamp(iou, min=0, max=1)

        pred_erase = decoded_pred[fg_proposal_erase].unsqueeze(0)       # (1, N, 7)
        target_erase = decoded_target[fg_proposal_erase].unsqueeze(0)   # (1, N, 7)
        pred_erase[:,:,[3,4]] = pred_erase[:,:,[4,3]]
        target_erase[:,:,[3,4]] = target_erase[:,:,[4,3]]
        if fg_proposal_erase.sum() == 0:
            iou_erase = 0
        else:
            iou_erase = diff_iou_rotated_3d(pred_erase, target_erase.float())[0].squeeze(0)
            iou_erase = torch.clamp(iou_erase, min=0, max=1)

        # loss 
        lamb = 0.00005
        shift_fg_loss = torch.sum(- torch.log(1 - pred_cls[fg_proposal_shift] + 1e-6) * iou)
        # shift_fg_loss = 0
        # erase_fg_loss = torch.sum(- torch.log(1 - pred_cls[fg_proposal_erase] + 1e-6) * iou_erase)
        erase_fg_loss = 0
        fg_loss = shift_fg_loss + erase_fg_loss
        # bg_loss = torch.sum(- (1-pred_cls[bg_proposal]).pow(self.gamma) * torch.log(pred_cls[bg_proposal] + 1e-6))
        bg_loss = torch.zeros([1]).to(fg_loss.device)
        total_loss = fg_loss + lamb * bg_loss

        return total_loss, fg_proposal.sum()


    def loss(self,
            result, # backbone的输出
            anchors, # (1, H, W, 2, 7)
            reg_targets, # (B, H, W, 14) 
            labels, # (B, H, W, 2)
            attack_target, # pred / gt 
            erase_index,
            iteration,
            num
            ):
        
        # 生成两个结果
        spatial_features_2d = result
        pred_cls = self.cls_head(spatial_features_2d) # (B, 2, H, W)
        pred_loc = self.reg_head(spatial_features_2d) # (B, 14, H, W)

        _, bbox_temp = self.generate_predicted_boxes(pred_cls, pred_loc)
        pred_loc = bbox_temp.reshape(1, 100, 252, 7).contiguous()

        # 调整shape
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().squeeze(0)                # (H, W, 2)
        pred_cls = pred_cls.view(-1, )  
        pred_cls = torch.sigmoid(pred_cls)                                             # (2*H*W, 1)
        anchors = anchors.squeeze(0)                                                   # (H, W, 2, 7)
        
        # 先生成真实的pred，再选出foreground
        '''
        delta_to_boxes3d
        inputs: rm (B, H, W, 14), anchor (1, H, W, 2, 7)
        outputs: (B, H*W*2, 7)
        '''
        # decoded_pred = VoxelPostprocessor.delta_to_boxes3d(pred_loc, anchors).view(-1, 7)
        if attack_target == 'gt':
            decoded_target = VoxelPostprocessor.delta_to_boxes3d(reg_targets, anchors).view(-1, 7)
        else:
            decoded_target = reg_targets.view(-1,7)
        decoded_pred = pred_loc.view(-1,7)

        # gt/pred不同之处在于label的处理
        if attack_target == 'gt':
            labels = labels.squeeze(0).view(-1)    # (H*W, )
            fg_proposal = labels == 1              # (H*W, )
            bg_proposal = labels == 0              # (H*W, )
        else:
            labels = labels.squeeze(0).view(-1, )
            labels = torch.sigmoid(labels)
            fg_proposal = labels > 0.5             # (H*W, )
            bg_proposal = labels <= 0.5           # (H*W, )  
    

        pred = decoded_pred[fg_proposal].unsqueeze(0)       # (1, N, 7)
        target = decoded_target[fg_proposal].unsqueeze(0)   # (1, N, 7)

        # compute IoU
        # pred中7个特性的顺序是 (x,y,z,h,w,l,alpha)
        pred[:,:,[3,4]] = pred[:,:,[4,3]]
        target[:,:,[3,4]] = target[:,:,[4,3]]
        # input (1, N, 7), 其中7个特性为(x,y,z,w,h,l,alpha) 
        # output (1, N)
        if fg_proposal.sum() == 0:
            iou = 0
        else:
            # tmp_pred = pred.detach().cpu()
            # tmp_target = target.detach().cpu()
            # iou = diff_iou_rotated_3d(tmp_pred, tmp_target.float())[0].squeeze(0)
            # iou = iou.cuda()
            iou = diff_iou_rotated_3d(pred, target.float())[0].squeeze(0)
            iou = torch.clamp(iou, min=0, max=1)

        # loss 
        lamb = 0.00005
        fg_loss = torch.sum(- torch.log(1 - pred_cls[fg_proposal] + 1e-6) * iou)
        # fg_loss_without_iou = torch.sum(- torch.log(1 - pred_cls[fg_proposal] + 1e-6))
        # fg_loss = fg_loss_without_iou
        # bg_loss = torch.sum(- (1-pred_cls[bg_proposal]).pow(self.gamma) * torch.log(pred_cls[bg_proposal] + 1e-6))
        bg_loss = torch.zeros([1]).to(fg_loss.device)
        total_loss = fg_loss + lamb * bg_loss

        return total_loss, fg_proposal.sum()


if __name__ == "__main__":
    pgd = PGD(torch.nn.Conv2d(1,1,1,1).cuda(),
                torch.nn.Conv2d(1,1,1,1).cuda(),
                torch.nn.Conv2d(1,1,1,1).cuda(), 
                eps=0.1, alpha=0.1, steps=40, 
                attack_mode='others',
                n_att=1, colla_attack=True, 
                noise_attack=False,
                random_start=True,
                project=True,
                save_path=None)

    result = torch.Tensor(np.load('/GPFS/data/shengyin/OpenCOOD-main/result.npy')).cuda()
    reg_target = torch.Tensor(np.load('/GPFS/data/shengyin/OpenCOOD-main/reg_target.npy')).cuda()
    labels = torch.Tensor(np.load('/GPFS/data/shengyin/OpenCOOD-main/labels.npy')).cuda()
    anchors = torch.Tensor(np.load('/GPFS/data/shengyin/OpenCOOD-main/anchors.npy')).cuda()
    
    print(pgd.loss_(result=result, reg_targets=reg_target, labels=labels, anchors=anchors))
    
