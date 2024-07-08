'''
/************************************************************************
 MIT License
 Copyright (c) 2021 AI4CE Lab@NYU, MediaBrain Group@SJTU
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *************************************************************************/
/**
 *  @file    CoDetModule.py
 *  @author  YIMING LI (https://roboticsyimingli.github.io/)
 *  @date    10/10/2021
 *  @version 1.0
 *
 *  @brief Co-det Modules of Collaborative BEV Detection
 *
 *  @section DESCRIPTION
 *
 *  This is official implementation for: NeurIPS 2021 Learning Distilled Collaboration Graph for Multi-Agent Perception
 *
 */
'''
import torch.nn as nn
from utils.detection_util import *
from utils.min_norm_solvers import MinNormSolver
import numpy

from .pgd import PGD
from .geometry_attack import ShiftBox, ShiftBoxPGD, RegionBasedShiftBoxPGD
from .rm_com_pair import rm_com_pair
# from .attack_detection import attack_detection
from .attack_detection.utils import label_attacker
import time

class FaFModule(object):
    def __init__(self, model, teacher, config, optimizer, criterion, kd_flag,
                 attack=False, com=True, attack_mode='self', eps=0.1, alpha=0.1,
                 proj=True, attack_target='pred', vis_path=None, eva_num=1, step=15, 
                 smooth=False, smooth_sigma=0.0, noise_attack=False, cli_args_for_attack=[]):
        self.MGDA = config.MGDA
        if self.MGDA:
            self.encoder = model[0]
            self.head = model[1]
            self.optimizer_encoder = optimizer[0]
            self.optimizer_head = optimizer[1]
            self.scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_encoder, milestones=[50, 100, 150, 200], gamma=0.5)
            self.scheduler_head = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_head, milestones=[50, 100, 150, 200], gamma=0.5)
            self.MGDA = config.MGDA
        else:
            self.model = model
            self.kd_flag = kd_flag
            if self.kd_flag == 1:
                self.teacher = teacher
                for k, v in self.teacher.named_parameters():
                    v.requires_grad = False  # fix parameters
            self.optimizer = optimizer
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
        self.criterion = criterion  # {'cls_loss','loc_loss'}

        self.out_seq_len = config.pred_len
        self.category_num = config.category_num
        self.code_size = config.box_code_size
        self.loss_scale = None

        self.code_type = config.code_type
        self.loss_type = config.loss_type
        self.pred_len = config.pred_len
        self.only_det = config.only_det

        self.history_result = None

        if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
            self.alpha = 1.
        elif self.code_type == 'faf':
            if self.loss_type == 'corner_loss':
                self.alpha = 1.
                if not self.only_det:
                    self.alpha = 1.
            else:
                self.alpha = 0.1
        self.config = config

        if isinstance(attack, bool) or attack == "TRUE":
            self.attack = attack or attack == "TRUE"
            if self.attack:
                self.attack_model = PGD(model, eps=eps, alpha=alpha, steps=step,
                                        attack_mode=attack_mode, noise_attack=noise_attack, 
                                        n_att=1, colla_attack=True,  # 兼容
                                        project=proj, vis_path=vis_path)
                self.attack_target = attack_target
                # self.eva_num = eva_num

        elif isinstance(attack, str):
            # yaml file 
            from omegaconf import OmegaConf
            attack_conf = OmegaConf.load(attack)
            cli_update = OmegaConf.from_dotlist(cli_args_for_attack)
            attack_conf = OmegaConf.merge(attack_conf, cli_update)
            if attack_conf.attack is not None:
                self.attack = True
                if "pgd" in attack_conf.attack:
                    self.attack_model = PGD(model, **attack_conf.attack.pgd, attack_config_file=attack)
                elif "shift" in attack_conf.attack:
                    self.attack_model = ShiftBox(model, **attack_conf.attack.shift, attack_config_file=attack)
                elif "shift_pgd" in attack_conf.attack:
                    self.attack_model = ShiftBoxPGD(model=model, **attack_conf.attack.shift_pgd, attack_config_file=attack)
                elif "region_shift_pgd" in attack_conf.attack:
                    self.attack_model = RegionBasedShiftBoxPGD(model=model, **attack_conf.attack.region_shift_pgd, attack_config_file=attack)
                self.attack_target = attack_conf.attack.attack_target
            else:
                self.attack = False
        
        
        if smooth:
            from .smooth_model import SmoothMedianNMS, DetectionsAcc
            self.smooth_model = SmoothMedianNMS(self.model.module, sigma=smooth_sigma, accumulator=DetectionsAcc(loc_bin_count=10))

    def resume(self, path):
        def map_func(storage, location):
            return storage.cuda()

        if os.path.isfile(path):
            if rank == 0:
                print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path, map_location=map_func)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

            ckpt_keys = set(checkpoint['state_dict'].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print('caution: missing keys from checkpoint {}: {}'.format(path, k))
        else:
            print("=> no checkpoint found at '{}'".format(path))

    def corner_loss(self, anchors, reg_loss_mask, reg_targets, pred_result):
        N = pred_result.shape[0]
        anchors = anchors.unsqueeze(-2).expand(anchors.shape[0], anchors.shape[1],
                                               anchors.shape[2], anchors.shape[3], reg_loss_mask.shape[-1], anchors.shape[-1])
        assigned_anchor = anchors[reg_loss_mask]
        assigned_target = reg_targets[reg_loss_mask]
        assigned_pred = pred_result[reg_loss_mask]
        # print(assigned_anchor.shape,assigned_pred.shape,assigned_target.shape)
        # exit()
        pred_decode = bev_box_decode_torch(assigned_pred, assigned_anchor)
        target_decode = bev_box_decode_torch(assigned_target, assigned_anchor)
        pred_corners = center_to_corner_box2d_torch(
            pred_decode[..., :2], pred_decode[..., 2:4], pred_decode[..., 4:])
        target_corners = center_to_corner_box2d_torch(
            target_decode[..., :2], target_decode[..., 2:4], target_decode[..., 4:])
        loss_loc = torch.sum(torch.norm(
            pred_corners-target_corners, dim=-1)) / N

        return loss_loc

    def loss_calculator(self, result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels=None, motion_mask=None):
        loss_num = 0
        # calculate loss
        weights = torch.Tensor([0.005, 1.0, 1.0, 1.0, 1.0]).cuda().double()
        loss_cls = torch.sum(self.criterion['cls'](result['cls'], labels)) / N
        loss_num += 1
        #loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N

        # Motion state
        if not motion_labels is None:
            loss_motion = torch.sum(self.criterion['cls'](
                result['state'], motion_labels)) / N
            loss_num += 1

        loss_mask_num = torch.nonzero(
            reg_loss_mask.view(-1, reg_loss_mask.shape[-1])).size(0)
        # print(loss_mask_num)
        # print(torch.sum(reg_targets[:,:,:,:,0][reg_loss_mask[:,:,:,:,2]]))

        if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
            target = reg_targets[reg_loss_mask].reshape(-1, 5, 2)
            flip_target = torch.stack(
                [target[:, 0], target[:, 3], target[:, 4], target[:, 1], target[:, 2]], dim=-2)
            pred = result['loc'][reg_loss_mask].reshape(-1, 5, 2)
            t = torch.sum(torch.norm(pred-target, dim=-1), dim=-1)
            f = torch.sum(torch.norm(pred-flip_target, dim=-1), dim=-1)
            loss_loc = torch.sum(torch.min(t, f)) / N

        elif self.code_type == 'faf':
            if self.loss_type == 'corner_loss':
                if self.only_det:
                    loss_loc = self.corner_loss(
                        anchors, reg_loss_mask, reg_targets, result['loc'])
                    loss_num += 1
                elif self.config.pred_type in ['motion', 'center']:

                    # only center/motion for pred

                    loss_loc_1 = self.corner_loss(anchors, reg_loss_mask[..., 0][..., [
                                                  0]], reg_targets[..., [0], :], result['loc'][..., [0], :])
                    pred_reg_loss_mask = reg_loss_mask[..., 1:, :]
                    if self.config.motion_state:
                        pred_reg_loss_mask = motion_mask  # mask out static object
                    loss_loc_2 = F.smooth_l1_loss(
                        result['loc'][..., 1:, :][pred_reg_loss_mask], reg_targets[..., 1:, :][pred_reg_loss_mask])
                    loss_loc = loss_loc_1 + loss_loc_2
                    loss_num += 2

                # corners for pred
                else:
                    loss_loc = self.corner_loss(
                        anchors, reg_loss_mask, reg_targets, result['loc'])
                    loss_num += 1
            else:

                loss_loc = F.smooth_l1_loss(
                    result['loc'][reg_loss_mask], reg_targets[reg_loss_mask])
                loss_num += 1

        if self.loss_scale is not None:
            if len(self.loss_scale) == 4:
                loss = self.loss_scale[0]*loss_cls + self.loss_scale[1]*loss_loc_1 + \
                    self.loss_scale[2]*loss_loc_2 + \
                    self.loss_scale[3]*loss_motion
            elif len(self.loss_scale) == 3:
                loss = self.loss_scale[0]*loss_cls + self.loss_scale[1] * \
                    loss_loc_1 + self.loss_scale[2]*loss_loc_2
            else:
                loss = self.loss_scale[0]*loss_cls + \
                    self.loss_scale[1]*loss_loc
        elif not motion_labels is None:
            loss = loss_cls + loss_loc + loss_motion
        else:
            loss = loss_cls + loss_loc

        if loss_num == 2:
            return (loss_num, loss, loss_cls, loss_loc)
        elif loss_num == 3:
            return (loss_num, loss, loss_cls, loss_loc_1, loss_loc_2)
        elif loss_num == 4:
            return (loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion)

    def step(self, data, batch_size):

        bev_seq = data['bev_seq']
        labels = data['labels']
        reg_targets = data['reg_targets']
        reg_loss_mask = data['reg_loss_mask']
        anchors = data['anchors']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']

        # with torch.autograd.set_detect_anomaly(True):
        if self.config.flag.startswith('when2com') or self.config.flag.startswith('who2com'):
            if self.config.split == 'train':
                result = self.model(
                    bev_seq, trans_matrices, num_agent, batch_size=batch_size, training=True)
            else:
                result = self.model(bev_seq, trans_matrices, num_agent,
                                    batch_size=batch_size, inference=self.config.inference, training=False)
        else:
            if self.kd_flag == 1:
                result, x_8, x_7, x_6, x_5, fused_layer = self.model(
                    bev_seq, trans_matrices, num_agent, batch_size=batch_size)
            else:
                result = self.model(bev_seq, trans_matrices,
                                    num_agent, batch_size=batch_size)

        if self.kd_flag == 1:
            bev_seq_teacher = data['bev_seq_teacher']
            kd_weight = data['kd_weight']
            x_8_teacher, x_7_teacher, x_6_teacher, x_5_teacher, x_3_teacher, x_2_teacher = self.teacher(
                bev_seq_teacher)

            # for k, v in self.teacher.named_parameters():
            # 	if k != 'xxx.weight' and k != 'xxx.bias':
            # 		print(v.requires_grad)  # should be False

            # for k, v in self.model.named_parameters():
            # 	if k != 'xxx.weight' and k != 'xxx.bias':
            # 		print(v.requires_grad)  # should be False

            # -------- KD loss---------#
            kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

            # target_x8 = x_8_teacher.permute(0, 2, 3, 1).reshape(5 * batch_size * 256 * 256, -1)
            # student_x8 = x_8.permute(0, 2, 3, 1).reshape(5 * batch_size * 256 * 256, -1)
            # kd_loss_x8 = kl_loss_mean(F.log_softmax(student_x8, dim=1), F.softmax(target_x8, dim=1))
            # #
            target_x7 = x_7_teacher.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 128 * 128, -1)
            student_x7 = x_7.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 128 * 128, -1)
            kd_loss_x7 = kl_loss_mean(F.log_softmax(
                student_x7, dim=1), F.softmax(target_x7, dim=1))
            #
            target_x6 = x_6_teacher.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 64 * 64, -1)
            student_x6 = x_6.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 64 * 64, -1)
            kd_loss_x6 = kl_loss_mean(F.log_softmax(
                student_x6, dim=1), F.softmax(target_x6, dim=1))
            # #
            target_x5 = x_5_teacher.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 32 * 32, -1)
            student_x5 = x_5.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 32 * 32, -1)
            kd_loss_x5 = kl_loss_mean(F.log_softmax(
                student_x5, dim=1), F.softmax(target_x5, dim=1))

            target_x3 = x_3_teacher.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 32 * 32, -1)
            student_x3 = fused_layer.permute(
                0, 2, 3, 1).reshape(5 * batch_size * 32 * 32, -1)
            kd_loss_fused_layer = kl_loss_mean(F.log_softmax(
                student_x3, dim=1), F.softmax(target_x3, dim=1))

            kd_loss = kd_weight * \
                (kd_loss_x7 + kd_loss_x6 + kd_loss_x5 + kd_loss_fused_layer)
            # print(kd_loss)

        else:
            kd_loss = 0

        labels = labels.view(
            result['cls'].shape[0], -1, result['cls'].shape[-1])

        N = bev_seq.shape[0]

        loss_collect = self.loss_calculator(
            result, anchors, reg_loss_mask, reg_targets, labels, N)
        loss_num = loss_collect[0]
        if loss_num == 3:
            loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
        elif loss_num == 2:
            loss_num, loss, loss_cls, loss_loc = loss_collect
        elif loss_num == 4:
            loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

        loss = loss + kd_loss

        if self.MGDA:
            self.optimizer_encoder.zero_grad()
            self.optimizer_head.zero_grad()
            loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_head.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.config.pred_type in ['motion', 'center'] and not self.only_det:
            if self.config.motion_state:
                return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), loss_motion.item()
            else:
                return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item()
        else:
            return loss.item(), loss_cls.item(), loss_loc.item()

    def predict_all(self, data, batch_size, validation=True):
        NUM_AGENT = 5
        bev_seq = data['bev_seq']
        vis_maps = data['vis_maps']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']
        num_sensor = num_agent[0, 0]

        if self.config.flag.startswith('when2com') or self.config.flag.startswith('who2com'):
            if self.config.split == 'train':
                result = self.model(
                    bev_seq, trans_matrices, num_agent, batch_size=batch_size, training=True)
            else:
                result = self.model(bev_seq, trans_matrices, num_agent,
                                    batch_size=batch_size, inference=self.config.inference, training=False)
        else:
            result = self.model(bev_seq, trans_matrices,
                                num_agent, batch_size=batch_size)
        N = bev_seq.shape[0]

        if validation:
            labels = data['labels']
            anchors = data['anchors']
            reg_targets = data['reg_targets']
            reg_loss_mask = data['reg_loss_mask']
            motion_labels = None
            motion_mask = None

            labels = labels.view(
                result['cls'].shape[0], -1, result['cls'].shape[-1])

            if self.attack:
                if self.attack_model.attack_mode == 'self':
                    ref_result = self.model.module.forward_no_com(
                        bev_seq, trans_matrices, num_agent, batch_size=batch_size)
                else:
                    ref_result = result

                if self.attack_target == 'gt':
                    att_reg_targets = reg_targets
                    att_labels = labels
                elif self.attack_target == 'pred':
                    att_reg_targets = ref_result['loc']
                    att_labels = ref_result['cls']
                else:
                    raise NotImplementedError(self.attack_target)

                evasion = self.attack_model(bev_seq, trans_matrices, num_agent,
                                            batch_size, anchors, reg_loss_mask, reg_targets=att_reg_targets, labels=att_labels)
                result = self.attack_model.inference(
                    bev_seq, evasion, trans_matrices, num_agent, batch_size=batch_size)

            if self.config.motion_state:
                motion_labels = data['motion_label']
                motion_mask = data['motion_mask']
                motion_labels = motion_labels.view(
                    result['state'].shape[0], -1, result['state'].shape[-1])
            N = bev_seq.shape[0]

            loss_collect = self.loss_calculator(
                result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels, motion_mask)
            loss_num = loss_collect[0]
            if loss_num == 3:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
            elif loss_num == 2:
                loss_num, loss, loss_cls, loss_loc = loss_collect
            elif loss_num == 4:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

        seq_results = [[] for i in range(NUM_AGENT)]

        for k in range(NUM_AGENT):
            bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

            if torch.nonzero(bev_seq).shape[0] == 0:
                seq_results[k] = []
            else:
                batch_box_preds = torch.unsqueeze(
                    result['loc'][k, :, :, :, :, :], 0)
                batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
                anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :], 0)
                batch_motion_preds = None

                if not self.only_det:
                    if self.config.pred_type == 'center':
                        batch_box_preds[:, :, :, :, 1:,
                                        2:] = batch_box_preds[:, :, :, :, [0], 2:]

                class_selected = apply_nms_det_detectron2(
                    batch_box_preds, batch_cls_preds, anchors, self.code_type, self.config, batch_motion_preds)
                seq_results[k] = class_selected

        if validation:
            return loss.item(), loss_cls.item(), loss_loc.item(), seq_results
        else:
            return seq_results

    def attack_transfer(self, data, batch_size, validation=True):
        NUM_AGENT = 5
        bev_seq = data['bev_seq']
        vis_maps = data['vis_maps']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']
        num_sensor = num_agent[0, 0]

        if self.config.flag.startswith('when2com') or self.config.flag.startswith('who2com'):
            if self.config.split == 'train':
                result = self.model(
                    bev_seq, trans_matrices, num_agent, batch_size=batch_size, training=True)
            else:
                result = self.model(bev_seq, trans_matrices, num_agent,
                                    batch_size=batch_size, inference=self.config.inference, training=False)
        else:
            result = self.model(bev_seq, trans_matrices,
                                num_agent, batch_size=batch_size)
        N = bev_seq.shape[0]

        if validation:
            labels = data['labels']
            anchors = data['anchors']
            reg_targets = data['reg_targets']
            reg_loss_mask = data['reg_loss_mask']
            motion_labels = None
            motion_mask = None

            labels = labels.view(
                result['cls'].shape[0], -1, result['cls'].shape[-1])

            if self.attack:
                if not self.attack_model.com:
                    ref_result = self.model.module.forward_no_com(
                        bev_seq, trans_matrices, num_agent, batch_size=batch_size)
                else:
                    ref_result = result

                if self.attack_target == 'gt':
                    att_reg_targets = reg_targets
                    att_labels = labels
                elif self.attack_target == 'pred':
                    att_reg_targets = ref_result['loc']
                    att_labels = ref_result['cls']
                else:
                    raise NotImplementedError(self.attack_target)

                evasion = self.attack_model(bev_seq, trans_matrices, num_agent,
                                            batch_size, anchors, reg_loss_mask, reg_targets=att_reg_targets, labels=att_labels)
                result = self.attack_model.inference(
                    bev_seq, evasion, trans_matrices, num_agent, batch_size=batch_size)

                #########################
                # transfer attack to broadcast
                #########################
                # att_src, att_tgt = self.attack_model.get_attack_pair(num_agent)
                # evasion, att_src, att_tgt = self.attack_model.attack_transfer(evasion, num_agent, att_src, att_tgt)

                # result = self.model(bevs=bev_seq, trans_matrices=trans_matrices, batch_size=batch_size, num_agent_tensor=num_agent, attack=evasion, attack_src=att_src, attack_tgt=att_tgt)

                #########################
                # transfer attack to self
                #########################
                # att_src, att_tgt = self.attack_model.get_attack_pair(num_agent)
                # att_tgt = att_src

                # com_src, com_tgt = self.model.module.get_com_pair(num_agent, n_com=1)
                # result = self.model(bevs=bev_seq, trans_matrices=trans_matrices, batch_size=batch_size, num_agent_tensor=num_agent, com_src=com_src, com_tgt=com_tgt,attack=evasion, attack_src=att_src, attack_tgt=att_tgt)

            if self.config.motion_state:
                motion_labels = data['motion_label']
                motion_mask = data['motion_mask']
                motion_labels = motion_labels.view(
                    result['state'].shape[0], -1, result['state'].shape[-1])
            N = bev_seq.shape[0]

            loss_collect = self.loss_calculator(
                result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels, motion_mask)
            loss_num = loss_collect[0]
            if loss_num == 3:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
            elif loss_num == 2:
                loss_num, loss, loss_cls, loss_loc = loss_collect
            elif loss_num == 4:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

        seq_results = [[] for i in range(NUM_AGENT)]

        for k in range(NUM_AGENT):
            bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

            if torch.nonzero(bev_seq).shape[0] == 0:
                seq_results[k] = []
            else:
                batch_box_preds = torch.unsqueeze(
                    result['loc'][k, :, :, :, :, :], 0)
                batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
                anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :], 0)
                batch_motion_preds = None

                if not self.only_det:
                    if self.config.pred_type == 'center':
                        batch_box_preds[:, :, :, :, 1:,
                                        2:] = batch_box_preds[:, :, :, :, [0], 2:]

                class_selected = apply_nms_det_detectron2(
                    batch_box_preds, batch_cls_preds, anchors, self.code_type, self.config, batch_motion_preds)
                seq_results[k] = class_selected

        if validation:
            return loss.item(), loss_cls.item(), loss_loc.item(), seq_results
        else:
            return seq_results

    def predict_all_with_box_com(self, data, trans_matrices_map, validation=True):
        NUM_AGENT = 5
        bev_seq = data['bev_seq']
        vis_maps = data['vis_maps']
        trans_matrices = data['trans_matrices']
        num_agent_tensor = data['num_agent']
        num_sensor = num_agent_tensor[0, 0]

        if self.MGDA:
            x = self.encoder(bev_seq)
            result = self.head(x)
        else:
            result = self.model(bev_seq, trans_matrices,
                                num_agent_tensor, batch_size=1)

        N = bev_seq.shape[0]

        if validation:
            labels = data['labels']
            anchors = data['anchors']
            reg_targets = data['reg_targets']
            reg_loss_mask = data['reg_loss_mask']
            motion_labels = None
            motion_mask = None

            labels = labels.view(
                result['cls'].shape[0], -1, result['cls'].shape[-1])

            if self.config.motion_state:
                motion_labels = data['motion_label']
                motion_mask = data['motion_mask']
                motion_labels = motion_labels.view(
                    result['state'].shape[0], -1, result['state'].shape[-1])
            N = bev_seq.shape[0]

            loss_collect = self.loss_calculator(result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels,
                                                motion_mask)
            loss_num = loss_collect[0]
            if loss_num == 3:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
            elif loss_num == 2:
                loss_num, loss, loss_cls, loss_loc = loss_collect
            elif loss_num == 4:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

        seq_results = [[] for i in range(NUM_AGENT)]
        local_results_wo_local_nms = [[] for i in range(NUM_AGENT)]
        local_results_af_local_nms = [[] for i in range(NUM_AGENT)]

        global_points = [[] for i in range(num_sensor)]
        cls_preds = [[] for i in range(num_sensor)]
        global_boxes_af_localnms = [[] for i in range(num_sensor)]
        box_scores_af_localnms = [[] for i in range(num_sensor)]

        forward_message_size = 0
        forward_message_size_two_nms = 0

        for k in range(NUM_AGENT):
            bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

            if torch.nonzero(bev_seq).shape[0] == 0:
                seq_results[k] = []
            else:
                batch_box_preds = torch.unsqueeze(
                    result['loc'][k, :, :, :, :, :], 0)
                batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
                anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :], 0)

                if self.config.motion_state:
                    batch_motion_preds = result['state']
                else:
                    batch_motion_preds = None

                if not self.only_det:
                    if self.config.pred_type == 'center':
                        batch_box_preds[:, :, :, :, 1:,
                                        2:] = batch_box_preds[:, :, :, :, [0], 2:]

                class_selected, box_scores_pred_cls = apply_nms_det(batch_box_preds, batch_cls_preds, anchors,
                                                                    self.code_type, self.config, batch_motion_preds)

                # transform all the boxes before local nms to the global coordinate
                # global_points[k], cls_preds[k] = apply_box_global_transform(trans_matrices_map[k], batch_box_preds,
                #                                                            batch_cls_preds, anchors, self.code_type,
                #                                                            self.config, batch_motion_preds)

                # transform the boxes after local nms to the global coordinate
                global_boxes_af_localnms[k], box_scores_af_localnms[k] = apply_box_global_transform_af_localnms(
                    trans_matrices_map[k], class_selected, box_scores_pred_cls)
                # print(cls_preds[k].shape, box_scores_af_localnms[k].shape)

                forward_message_size = forward_message_size + 256 * 256 * 6 * 4 * 2
                forward_message_size_two_nms = forward_message_size_two_nms + global_boxes_af_localnms[k].shape[
                    0] * 4 * 2

        # global results with one NMS
        # all_points_scene = numpy.concatenate(tuple(global_points), 0)
        # cls_preds_scene = torch.cat(tuple(cls_preds), 0)
        # class_selected_global = apply_nms_global_scene(all_points_scene, cls_preds_scene)

        # global results with two NMS
        global_boxes_af_local_nms = numpy.concatenate(
            tuple(global_boxes_af_localnms), 0)
        box_scores_af_local_nms = torch.cat(tuple(box_scores_af_localnms), 0)
        class_selected_global_af_local_nms = apply_nms_global_scene(
            global_boxes_af_local_nms, box_scores_af_local_nms)

        # transform the consensus global boxes to local agents (two NMS)
        back_message_size_two_nms = 0
        for k in range(num_sensor):
            local_results_af_local_nms[k], ms = apply_box_local_transform(class_selected_global_af_local_nms,
                                                                          trans_matrices_map[k])
            back_message_size_two_nms = back_message_size_two_nms + ms

        sample_bandwidth_two_nms = forward_message_size_two_nms + back_message_size_two_nms

        # transform the consensus global boxes to local agents (One NMS)
        # back_message_size = 0
        # for k in range(num_sensor):
        #    local_results_wo_local_nms[k], ms = apply_box_local_transform(class_selected_global, trans_matrices_map[k])
        #    back_message_size = back_message_size + ms

        # sample_bandwidth = forward_message_size + back_message_size

        return loss.item(), loss_cls.item(), loss_loc.item(), local_results_af_local_nms, class_selected_global_af_local_nms, sample_bandwidth_two_nms

    def cal_loss_scale(self, data):
        bev_seq = data['bev_seq']
        labels = data['labels']
        reg_targets = data['reg_targets']
        reg_loss_mask = data['reg_loss_mask']
        anchors = data['anchors']
        motion_labels = None
        motion_mask = None

        with torch.no_grad():
            shared_feats = self.encoder(bev_seq)
        shared_feats_tensor = shared_feats.clone().detach().requires_grad_(True)
        result = self.head(shared_feats_tensor)
        if self.config.motion_state:
            motion_labels = data['motion_label']
            motion_mask = data['motion_mask']
            motion_labels = motion_labels.view(
                result['state'].shape[0], -1, result['state'].shape[-1])
        self.optimizer_encoder.zero_grad()
        self.optimizer_head.zero_grad()
        grads = {}
        labels = labels.view(
            result['cls'].shape[0], -1, result['cls'].shape[-1])
        N = bev_seq.shape[0]

        # calculate loss
        grad_len = 0

        '''
        Classification Loss
        '''
        loss_cls = self.alpha * \
            torch.sum(self.criterion['cls'](result['cls'], labels)) / N
        #loss_loc = torch.sum(self.criterion['loc'](result['loc'],reg_targets,mask = reg_loss_mask)) / N
        self.optimizer_encoder.zero_grad()
        self.optimizer_head.zero_grad()

        loss_cls.backward(retain_graph=True)
        grads[0] = []
        grads[0].append(shared_feats_tensor.grad.data.clone().detach())
        shared_feats_tensor.grad.data.zero_()
        grad_len += 1

        '''
        Localization Loss
        '''
        loc_scale = False
        loss_mask_num = torch.nonzero(
            reg_loss_mask.view(-1, reg_loss_mask.shape[-1])).size(0)

        if self.code_type in ['corner_1', 'corner_2', 'corner_3']:
            target = reg_targets[reg_loss_mask].reshape(-1, 5, 2)
            flip_target = torch.stack(
                [target[:, 0], target[:, 3], target[:, 4], target[:, 1], target[:, 2]], dim=-2)
            pred = result['loc'][reg_loss_mask].reshape(-1, 5, 2)
            t = torch.sum(torch.norm(pred-target, dim=-1), dim=-1)
            f = torch.sum(torch.norm(pred-flip_target, dim=-1), dim=-1)
            loss_loc = torch.sum(torch.min(t, f)) / N

        elif self.code_type == 'faf':
            if self.loss_type == 'corner_loss':
                if self.only_det:
                    loss_loc = self.corner_loss(
                        anchors, reg_loss_mask, reg_targets, result['loc'])
                elif self.config.pred_type in ['motion', 'center']:

                    # only center/motion for pred

                    loss_loc_1 = self.corner_loss(anchors, reg_loss_mask[..., 0][..., [
                                                  0]], reg_targets[..., [0], :], result['loc'][..., [0], :])
                    pred_reg_loss_mask = reg_loss_mask[..., 1:, :]
                    if self.config.motion_state:
                        pred_reg_loss_mask = motion_mask  # mask out static object
                    loss_loc_2 = F.smooth_l1_loss(
                        result['loc'][..., 1:, :][pred_reg_loss_mask], reg_targets[..., 1:, :][pred_reg_loss_mask])

                    self.optimizer_encoder.zero_grad()
                    self.optimizer_head.zero_grad()

                    loss_loc_1.backward(retain_graph=True)
                    grads[1] = []
                    grads[1].append(
                        shared_feats_tensor.grad.data.clone().detach())
                    shared_feats_tensor.grad.data.zero_()

                    self.optimizer_encoder.zero_grad()
                    self.optimizer_head.zero_grad()

                    loss_loc_2.backward(retain_graph=True)
                    grads[2] = []
                    grads[2].append(
                        shared_feats_tensor.grad.data.clone().detach())
                    shared_feats_tensor.grad.data.zero_()
                    loc_scale = True
                    grad_len += 2

                # corners for pred
                else:
                    loss_loc = self.corner_loss(
                        anchors, reg_loss_mask, reg_targets, result['loc'])
            else:

                loss_loc = F.smooth_l1_loss(
                    result['loc'][reg_loss_mask], reg_targets[reg_loss_mask])

            if not loc_scale:
                grad_len += 1
                self.optimizer_encoder.zero_grad()
                self.optimizer_head.zero_grad()
                loss_loc.backward(retain_graph=True)
                grads[1] = []
                grads[1].append(shared_feats_tensor.grad.data.clone().detach())
                shared_feats_tensor.grad.data.zero_()

        '''
        Motion state Loss
        '''
        if self.config.motion_state:
            loss_motion = torch.sum(self.criterion['cls'](
                result['state'], motion_labels)) / N

            self.optimizer_encoder.zero_grad()
            self.optimizer_head.zero_grad()

            loss_motion.backward(retain_graph=True)
            grads[3] = []
            grads[3].append(shared_feats_tensor.grad.data.clone().detach())
            shared_feats_tensor.grad.data.zero_()
            grad_len += 1

        # ---------------------------------------------------------------------
        # -- Frank-Wolfe iteration to compute scales.
        scale = np.zeros(grad_len, dtype=np.float32)
        sol, min_norm = MinNormSolver.find_min_norm_element(
            [grads[t] for t in range(grad_len)])
        for i in range(grad_len):
            scale[i] = float(sol[i])

        # print(scale)
        return scale

    def step_with_at(self, data, batch_size, trace=False):
        # import ipdb;ipdb.set_trace()
        bev_seq = data['bev_seq']
        labels = data['labels']
        reg_targets = data['reg_targets']
        reg_loss_mask = data['reg_loss_mask']
        anchors = data['anchors']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']

        gt_bbox = data['gt_bbox']

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        if self.config.flag.startswith('when2com') or self.config.flag.startswith('who2com'):
            raise NotImplementedError()
            if self.config.split == 'train':
                result = self.model(
                    bev_seq, trans_matrices, num_agent, batch_size=batch_size, training=True)
            else:
                result = self.model(bev_seq, trans_matrices, num_agent,
                                    batch_size=batch_size, inference=self.config.inference, training=False)
        else:
            if self.kd_flag == 1:
                # generate perturbation
                com_src, com_tgt = model.get_default_com_pair(num_agent)
                
                if self.attack:
                    model.eval()
                    if self.attack_target == "gt":
                        # use gt as attack label
                        if isinstance(self.attack_model, PGD):
                            attack = self.attack_model(bev_seq, trans_matrices, num_agent, batch_size, 
                                    anchors, reg_loss_mask, reg_targets, labels, com_src, com_tgt)
                        elif isinstance(self.attack_model, ShiftBox):
                            if isinstance(gt_bbox[0], list):
                                gt_bbox = [[b['bboxes'].reshape(-1, 4, 2) for b in b_list] for b_list in gt_bbox]
                            else:
                                gt_bbox = [b['bboxes'].reshape(-1, 4, 2) for b in gt_bbox]
                            attack = self.attack_model(bev_seq, trans_matrices, num_agent, batch_size, anchors, reg_loss_mask, reg_targets, labels, gt_bbox, None, com_src, com_tgt)
                    elif self.attack_target == "pred":
                        # use prediction as attack labe
                        with torch.no_grad():
                            result = model(bev_seq, trans_matrices, num_agent, batch_size=batch_size)[0]
                        attack = self.attack_model(bev_seq, trans_matrices, num_agent, batch_size, 
                                anchors, reg_loss_mask, result['loc'], result['cls'], com_src, com_tgt)
                    
                    attack_src, attack_tgt = self.attack_model.get_attack_pair(num_agent)
                    model.train()
                else:
                    attack, attack_src, attack_tgt = None, None, None

                result, x_8, x_7, x_6, x_5, fused_layer = self.model(
                    bev_seq, trans_matrices, num_agent, batch_size=batch_size, attack=attack, attack_src=attack_src, attack_tgt=attack_tgt)

                if trace:
                    import ipdb;ipdb.set_trace()
            else:
                result = self.model(bev_seq, trans_matrices,
                                    num_agent, batch_size=batch_size)

        if self.kd_flag == 1:
            bev_seq_teacher = data['bev_seq_teacher']
            kd_weight = data['kd_weight']
            x_8_teacher, x_7_teacher, x_6_teacher, x_5_teacher, x_3_teacher, x_2_teacher = self.teacher(
                bev_seq_teacher)

            # -------- KD loss---------#
            kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

            target_x7 = x_7_teacher.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 128 * 128, -1)
            student_x7 = x_7.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 128 * 128, -1)
            kd_loss_x7 = kl_loss_mean(F.log_softmax(
                student_x7, dim=1), F.softmax(target_x7, dim=1))
            
            target_x6 = x_6_teacher.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 64 * 64, -1)
            student_x6 = x_6.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 64 * 64, -1)
            kd_loss_x6 = kl_loss_mean(F.log_softmax(
                student_x6, dim=1), F.softmax(target_x6, dim=1))
            
            target_x5 = x_5_teacher.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 32 * 32, -1)
            student_x5 = x_5.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 32 * 32, -1)
            kd_loss_x5 = kl_loss_mean(F.log_softmax(
                student_x5, dim=1), F.softmax(target_x5, dim=1))

            target_x3 = x_3_teacher.permute(0, 2, 3, 1).reshape(
                5 * batch_size * 32 * 32, -1)
            student_x3 = fused_layer.permute(
                0, 2, 3, 1).reshape(5 * batch_size * 32 * 32, -1)
            kd_loss_fused_layer = kl_loss_mean(F.log_softmax(
                student_x3, dim=1), F.softmax(target_x3, dim=1))

            kd_loss = kd_weight * \
                (kd_loss_x7 + kd_loss_x6 + kd_loss_x5 + kd_loss_fused_layer)
            # print(kd_loss)

        else:
            kd_loss = 0

        labels = labels.view(
            result['cls'].shape[0], -1, result['cls'].shape[-1])

        N = bev_seq.shape[0]

        loss_collect = self.loss_calculator(
            result, anchors, reg_loss_mask, reg_targets, labels, N)
        loss_num = loss_collect[0]
        if loss_num == 3:
            loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
        elif loss_num == 2:
            loss_num, loss, loss_cls, loss_loc = loss_collect
        elif loss_num == 4:
            loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

        loss = loss + kd_loss

        if self.MGDA:
            self.optimizer_encoder.zero_grad()
            self.optimizer_head.zero_grad()
            loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_head.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.config.pred_type in ['motion', 'center'] and not self.only_det:
            if self.config.motion_state:
                return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item(), loss_motion.item()
            else:
                return loss.item(), loss_cls.item(), loss_loc_1.item(), loss_loc_2.item()
        else:
            return loss.item(), loss_cls.item(), loss_loc.item()

    @torch.no_grad()
    def attack_detection(self, data, batch_size, detection_method="none", load_score=None,
                         load_attack=True, save_attack=False, robosac_cfg = None, match_para = 1, multi_test_alpha = 0.05, cnt = 0):
        
        # Inputs
        assert batch_size == 1, 'batch_size should be 1'
        NUM_AGENT = 5
        bev_seq = data['bev_seq']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']

        labels = data['labels']
        anchors = data['anchors']
        reg_targets = data['reg_targets']
        reg_loss_mask = data['reg_loss_mask']
        scene_name = data['scene_name']

        gt_bbox = data['gt_bbox']

        if self.attack:
            attack_config_file_path = os.path.abspath(self.attack_model.attack_config_file).split("/")
            save_file_name = "/".join(attack_config_file_path[attack_config_file_path.index("attack")+1:])
            save_file_name = save_file_name.replace("yaml", "npz")
        # import ipdb;ipdb.set_trace()
        additional_results = {}

        # model (donnot support DP for now)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        com_src, com_tgt = model.get_default_com_pair(num_agent)
        # attacker_labels = torch.zeros(com_src.shape[0]).long().to(com_src.device)
        load = self.attack and load_attack and os.path.exists(os.path.join("saved_attacks", scene_name, save_file_name))

        if self.attack and not load:
            # generate perturbation
            if self.attack_target == "gt":
                # use gt as attack label
                if isinstance(self.attack_model, PGD):
                    attack = self.attack_model(bev_seq, trans_matrices, num_agent, batch_size, 
                            anchors, reg_loss_mask, reg_targets, labels, com_src, com_tgt)
                elif isinstance(self.attack_model, ShiftBox):
                    if isinstance(gt_bbox[0], list):
                        gt_bbox = [[b['bboxes'].reshape(-1, 4, 2) for b in b_list] for b_list in gt_bbox]
                    else:
                        gt_bbox = [b['bboxes'].reshape(-1, 4, 2) for b in gt_bbox]
                    attack = self.attack_model(bev_seq, trans_matrices, num_agent, batch_size, anchors, reg_loss_mask, reg_targets, labels, gt_bbox, None, com_src, com_tgt)
                
            elif self.attack_target == "pred":
                # import ipdb;ipdb.set_trace()
                # use prediction as attack label
                if isinstance(self.attack_model, PGD):
                    with torch.no_grad():
                        result = model(bev_seq, trans_matrices, num_agent, batch_size=batch_size)
                    attack = self.attack_model(bev_seq, trans_matrices, num_agent, batch_size, 
                            anchors, reg_loss_mask, result['loc'], result['cls'], com_src, com_tgt)
                elif isinstance(self.attack_model, ShiftBox):
                    with torch.no_grad():
                        result, pred_bbox = model(bev_seq, trans_matrices, num_agent, batch_size=batch_size, nms=True, batch_anchors=anchors)
                    pred_bbox = [b[0]['pred'].squeeze(1) for b in pred_bbox]
                    # FIXME hack, only support batch_size=1
                    assert batch_size == 1, "this is a hack, only support batch_size==1"
                    pred_bbox = [pred_bbox]
                    attack = self.attack_model(bev_seq, trans_matrices, num_agent, batch_size, anchors, reg_loss_mask, result['loc'], result['cls'], pred_bbox, None, com_src, com_tgt)
                
            elif self.attack_target == "none":
                attack = None  # FIXME bug during multi-com-forward
            
            attack_src, attack_tgt = self.attack_model.get_attack_pair(num_agent)
            # attacker_labels = label_attacker(attacker_labels, com_src, com_tgt, attack_src, attack_tgt)

        elif self.attack and load:
            attack = np.load(os.path.join("saved_attacks", scene_name, save_file_name))['arr_0']
            # if (attack == None).all():
            #     attack = None
            # else:
            attack = torch.Tensor(attack).to(bev_seq.device)
            attack_src, attack_tgt = self.attack_model.get_attack_pair(num_agent)
        else:
            # no attack
            attack = None
            attack_src = None
            attack_tgt = None

        # import ipdb;ipdb.set_trace()
        # 检查多个attacker攻击是否会比单个attacker更差。在2个attacker中只保留一个进行攻击
        # if len(attack) > 0:
        #     n = num_agent[0, 0]
        #     attack = attack[:n]
        #     attack_src = attack_src[:n]
        #     attack_tgt = attack_tgt[:n]
        
        # import ipdb;ipdb.set_trace()
        if self.attack and save_attack and not load:
            # only support batch_size = 1
            attack_config_file_path = os.path.abspath(self.attack_model.attack_config_file).split("/")
            save_file_name = "/".join(attack_config_file_path[attack_config_file_path.index("attack")+1:])
            save_file_name = save_file_name.replace("yaml", "npz")
            if os.path.exists(os.path.join("saved_attacks", scene_name, save_file_name)):
                print(f"{save_file_name} exists")
            elif not os.path.exists(os.path.dirname(os.path.join("saved_attacks", scene_name, save_file_name))):
                os.makedirs(os.path.dirname(os.path.join("saved_attacks", scene_name, save_file_name)))
            np_attack = attack.detach().cpu().numpy()
            np.savez_compressed(os.path.join("saved_attacks", scene_name, save_file_name), np_attack)
            

        if detection_method == "match_cost":
            # inference for attack detection
            com_srcs, com_tgts = model.get_attack_det_com_pairs(num_agent)
            results_list = model.multi_com_forward(
                bev_seq, trans_matrices, com_srcs, com_tgts, attack, attack_src, attack_tgt, batch_size)
            
            com_srcs_to_det = torch.cat([com_srcs[i][1::2] for i in range(1, len(com_srcs))], dim=0)
            com_tgts_to_det = torch.cat([com_tgts[i][1::2] for i in range(1, len(com_tgts))], dim=0)
            attacker_labels = label_attacker(com_srcs_to_det, com_tgts_to_det, attack_src, attack_tgt)
            # --------- start ---------------
            # single victim inference
            # if len(attack_tgt) > 0:
            #     selected_attack_index = attack_tgt[:, 1] == attack_tgt[0, 1]
            #     attack = attack[selected_attack_index]
            #     attack_src = attack_src[selected_attack_index]
            #     attack_tgt = attack_tgt[selected_attack_index]
            # else:
            #     attack = None
            #     attack_src = None
            #     attack_tgt = None

            
            # results_list = model.single_view_multi_com_forward(
            #     bev_seq, trans_matrices, com_srcs, com_tgts, attack, attack_src, attack_tgt, batch_size
            # )
            # ----------- end -----------------

            k = num_agent[0, 0]
            box_list = [model.post_process(results, anchors, k) for results in results_list]

            ego_result = box_list[0]
            match_cost = [model.matcher(box_list[i], ego_result) for i in range(1, len(box_list))]
            # remove attacker according to match_cost
            # modify com_src, com_tgt here 
            # TODO determine threshold by no attack scenarios `experiments/no_attack`
            percentiles = {
            50: 0.536744492804809,
            60: 0.5839201450420946,
            70: 0.6304444880969823,
            80: 0.6880368602735208,
            90: 0.7671696103588593,
            95: 0.8326803338450066,}
            # 思考：the normal agent with high match cost is whether informative or noisy?
            match_costs_tensor = torch.Tensor(match_cost).cuda()
            match_costs_tensor = match_costs_tensor[:, 0] if match_costs_tensor.ndim == 3 else match_costs_tensor

            # detected_src_relative_index, detected_tgt = torch.where(match_costs_tensor > percentiles[95])
            # detected_src = (detected_src_relative_index + detected_tgt + 1) % k

            match_costs_tensor = match_costs_tensor.reshape(-1)
            # assert (com_srcs_to_det[match_costs_tensor > percentiles[95], 1] == detected_src).all()
            # assert (com_tgts_to_det[match_costs_tensor > percentiles[95], 1] == detected_tgt).all()

            com_src, com_tgt = model.get_default_com_pair(num_agent)
            # com_src, com_tgt = model.remove_com_pair(com_src, com_tgt, detected_src, detected_tgt)
            is_attacker = match_costs_tensor > percentiles[95]
            detected_src = com_srcs_to_det[is_attacker]
            detected_tgt = com_tgts_to_det[is_attacker]

            total = len(attacker_labels)
            correct = (attacker_labels == is_attacker).sum().item()
            com_src, com_tgt = rm_com_pair(com_src, com_tgt, detected_src, detected_tgt)

            additional_results['bboxes'] = box_list
            additional_results['match_cost'] = match_cost
            additional_results['total'] = total
            additional_results['correct'] = correct
        
        elif detection_method == "gt":
            if len(attack_src) > 0:
                com_src, com_tgt = rm_com_pair(com_src, com_tgt, attack_src, attack_tgt)
        
        elif detection_method == "binary_classifier":
            if not hasattr(self, "binary_classifier"):
                from utils.binary_classifier import my_CNN as BinaryClassifier
                self.binary_classifier = BinaryClassifier("/DB/data/shengyin/classifer_of_adverisial/parameter_in_epoch_30.pth").cuda()
                self.binary_classifier.eval()

            # generate perturbed feature map
            com_src, com_tgt = model.get_default_com_pair(num_agent)
            features = model.encode(bev_seq)
            com_features = features[3]
            B, C, H, W = com_features.shape
            com_features = com_features.reshape(-1, model.agent_num, C, H, W)
            src_features = com_features[com_src[:, 0],
                                        com_src[:, 1]]
            
            src_features_with_perturbation = src_features + \
                model.place_attack_v2(attack, attack_src, attack_tgt, com_src, com_tgt) \
                    if attack is not None else src_features
            src_features_with_perturbation = src_features_with_perturbation.detach()
            score = self.binary_classifier(src_features_with_perturbation)  # 0 is perturbed, 1 is normal
            keep_index = score.argmax(dim=1) == 1 
            self_loop = com_src[:, 1] == com_tgt[:, 1]
            keep_index = keep_index | self_loop

            com_src = com_src[keep_index]
            com_tgt = com_tgt[keep_index]
            # import ipdb;ipdb.set_trace()
        elif detection_method == "raw_autoencoder":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.raw_autoencoder import RawAutoEncoderDetector
                self.attacker_detection_model = RawAutoEncoderDetector()
                self.attacker_detection_model.to(bev_seq.device)
            
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)
        
        elif detection_method == "residual_autoencoder":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.residual_autoencoder import ResidualAutoEncoderDetector
                self.attacker_detection_model = ResidualAutoEncoderDetector()
                self.attacker_detection_model.to(bev_seq.device)
            
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)
        
        elif detection_method == "residual_autoencoder_v0":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.residual_autoencoder_v0 import ResidualAutoEncoderV0Detector
                self.attacker_detection_model = ResidualAutoEncoderV0Detector()
                self.attacker_detection_model.to(bev_seq.device)
            
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)
        
        elif detection_method == "residual_autoencoder_v2":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.residual_autoencoder_v2 import ResidualAutoEncoderV2Detector
                self.attacker_detection_model = ResidualAutoEncoderV2Detector()
                self.attacker_detection_model.to(bev_seq.device)
            
            time1 = time.time()
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)
            
            spent_time = time.time() - time1
            additional_results['spent_time'] = spent_time

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)

        elif detection_method == "residual_autoencoder_v3":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.residual_autoencoder_v3 import ResidualAutoEncoderV3Detector
                self.attacker_detection_model = ResidualAutoEncoderV3Detector()
                self.attacker_detection_model.to(bev_seq.device)
            
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)
        
        elif detection_method == "match_cost_v2":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.match_cost import MatchCostDetector, LOAD_SCORE_BASE
                if load_score == None:
                    tmp_load_score = None
                else:
                    tmp_load_score = os.path.join(LOAD_SCORE_BASE, load_score)
                self.attacker_detection_model = MatchCostDetector(load_score=tmp_load_score, match_para = match_para)
                self.attacker_detection_model.to(bev_seq.device)

            time1 = time.time()
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size, cnt = cnt)
            
            spent_time = time.time() - time1
            additional_results['spent_time'] = spent_time
            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)

        elif detection_method == 'robosac':

            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.robosac import ROBOSAC_Detector
                # import pdb; pdb.set_trace()
                if robosac_cfg is None:
                    self.attacker_detection_model = ROBOSAC_Detector()
                else:
                    from omegaconf import OmegaConf
                    robosac_conf = OmegaConf.load(robosac_cfg)
                    self.attacker_detection_model = ROBOSAC_Detector(**robosac_conf)

                self.attacker_detection_model.to(bev_seq.device)

            time1 = time.time()
            com_src, com_tgt, robo_ego_result = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size, self.history_result)
            spent_time = time.time() - time1

            self.history_result = robo_ego_result

            additional_results['total'] = 0
            additional_results['correct'] = 0
            additional_results['spent_time'] = spent_time
            additional_results.update({})


        elif detection_method == "multi-test":
            # match_costs, reconstruction_loss = \
            # com_src, com_tgt = \
            # attack_detection(model, bev_seq, trans_matrices, num_agent, anchors, 
            #     attack, attack_src, attack_tgt, batch_size)
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.multi_test import MultiTestDetector
                self.attacker_detection_model = MultiTestDetector()
                self.attacker_detection_model.to(bev_seq.device)
            
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)
        
        elif detection_method == "multi-test-raev2":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.multi_test_load import MultiTestDetector, LOAD_DICT_BASE
                self.attacker_detection_model = MultiTestDetector(load_score={k: os.path.join(v, load_score) for k, v in LOAD_DICT_BASE.items()})
                self.attacker_detection_model.to(bev_seq.device)
            
            time1 = time.time()
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)
            
            spent_time = time.time() - time1
            additional_results['spent_time'] = spent_time

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)
        
        elif detection_method == "multi-test-v3":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.multi_test_v3 import MultiTestDetector, LOAD_DICT_BASE
                # import pdb; pdb.set_trace()
                if load_score == None:
                    tmp_load_score = None
                else:
                    tmp_load_score = {k: os.path.join(v, load_score) for k, v in LOAD_DICT_BASE.items()}
                self.attacker_detection_model = MultiTestDetector(load_score=tmp_load_score, multi_test_alpha=multi_test_alpha)
                self.attacker_detection_model.to(bev_seq.device)
            time1 = time.time()
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size, cnt = cnt)

            spent_time = time.time() - time1
            additional_results['spent_time'] = spent_time

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)

        elif detection_method == "none":
            com_src, com_tgt = model.get_default_com_pair(num_agent)
        elif detection_method == "check_ones":
            com_src, com_tgt = model.get_default_com_pair(num_agent)
            with torch.no_grad():
                result = model(bev_seq, trans_matrices, num_agent, batch_size=batch_size)
            pred_score = torch.softmax(result['cls'][reg_loss_mask.reshape(reg_loss_mask.shape[0], -1)], dim=-1)
            gt_ones = labels[reg_loss_mask.squeeze(-1)]
            pred_fg = (pred_score[:, 1] > 0.7).sum()
            gt_fg = (gt_ones[:, 1] == 1).sum()
            additional_results['pred_fg'] = pred_fg.item()
            additional_results['gt_fg'] = gt_fg.item()

        elif detection_method == "match_cost_unsupervised":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.match_cost_unsupervised import MatchCostUnsupervisedDetector, LOAD_SCORE_BASE
                self.attacker_detection_model = MatchCostUnsupervisedDetector(os.path.join(LOAD_SCORE_BASE, load_score))
                self.attacker_detection_model.to(bev_seq.device)
            
            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)

        elif detection_method == "match_cost_rm1":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.match_cost_rm1 import MatchCostDetectorRM1, LOAD_SCORE_BASE
                self.attacker_detection_model = MatchCostDetectorRM1(os.path.join(LOAD_SCORE_BASE, load_score))
                self.attacker_detection_model.to(bev_seq.device)

            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)

        elif detection_method == "kde":
            if not hasattr(self, "attacker_detection_model"):
                from .attack_detection.kde import KDEDetector
                self.attacker_detection_model = KDEDetector()
                self.attacker_detection_model.to(bev_seq.device)

            com_src, com_tgt, total, correct, appendix = self.attacker_detection_model(model, bev_seq, trans_matrices, num_agent, anchors, 
                attack, attack_src, attack_tgt, batch_size)

            additional_results['total'] = total
            additional_results['correct'] = correct
            additional_results.update(appendix)

        elif detection_method == 'oracle':

            com_src, com_tgt = model.get_default_com_pair(num_agent)
            tmp_com_src, tmp_com_tgt = [], []
            for com_idx in range(len(com_src)):
                tmp_src, tmp_tgt = com_src[com_idx].tolist(), com_tgt[com_idx].tolist()
                flag = False
                for idx in range(len(attack_src)):
                    src, tgt = attack_src[idx].tolist(), attack_tgt[idx].tolist()
                    if (src == tmp_src) and (tgt == tmp_tgt):
                        flag = True
                        break
                if not flag:
                    tmp_com_src.append(tmp_src)
                    tmp_com_tgt.append(tmp_tgt)
            
            com_src = torch.tensor(tmp_com_src).to(bev_seq.device)
            com_tgt = torch.tensor(tmp_com_tgt).to(bev_seq.device)

        elif detection_method == 'no_com':

            com_src = torch.tensor([[0,0],[0,1],[0,2],[0,3]]).to(bev_seq.device)
            com_tgt = torch.tensor([[0,0],[0,1],[0,2],[0,3]]).to(bev_seq.device)

        # final inference 
        final_result, final_box = model(bev_seq, trans_matrices, num_agent, batch_size=batch_size,
                                    com_src=com_src, com_tgt=com_tgt, 
                                    attack=attack, attack_src=attack_src, attack_tgt=attack_tgt,
                                    batch_anchors=anchors, nms=True, scene_name=scene_name)


        return final_result, final_box, additional_results

    def remove_attacker_inference(self, data, batch_size, validation=True):
        """
            1. 根据ground truth将attacker remove掉，作为baseline结果 [x]
            2. 根据attacker detection的结果，将attacker remove。[ ]
                - 需要考虑threshold的问题
                    - 对每个样本保留根据ground truth remove的结果，同时保存没有
                    remove的结果，根据score排序决定该样本是否remove，用类似AUC的
                    方式计算remove attacker的平均mAP
        """
        NUM_AGENT = 5
        bev_seq = data['bev_seq']
        vis_maps = data['vis_maps']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']
        num_sensor = num_agent[0, 0]

        if self.config.flag.startswith('when2com') or self.config.flag.startswith('who2com'):
            if self.config.split == 'train':
                result = self.model(
                    bev_seq, trans_matrices, num_agent, batch_size=batch_size, training=True)
            else:
                result = self.model(bev_seq, trans_matrices, num_agent,
                                    batch_size=batch_size, inference=self.config.inference, training=False)
        else:
            result = self.model(bev_seq, trans_matrices,
                                num_agent, batch_size=batch_size)
        N = bev_seq.shape[0]

        # com_src, com_tgt = self.model.module.get_default_com_pair(num_agent)
        com_src, com_tgt = self.model.module.get_com_pair(num_agent, 1)
        att_src, att_tgt = self.attack_model.get_attack_pair(num_agent)

        n_com_src, n_com_tgt = rm_com_pair(com_src, com_tgt, att_src, att_tgt)

        # import ipdb;ipdb.set_trace()

        removed_result = self.model(bev_seq, trans_matrices, num_agent, 
                                    batch_size=batch_size, com_src=n_com_src, com_tgt=n_com_tgt)

        result = removed_result  # FIXME rename this code
        if validation:
            labels = data['labels']
            anchors = data['anchors']
            reg_targets = data['reg_targets']
            reg_loss_mask = data['reg_loss_mask']
            motion_labels = None
            motion_mask = None

            labels = labels.view(
                result['cls'].shape[0], -1, result['cls'].shape[-1])

            if self.attack:
                if not self.attack_model.com:
                    ref_result = self.model.module.forward_no_com(
                        bev_seq, trans_matrices, num_agent, batch_size=batch_size)
                else:
                    ref_result = result

                if self.attack_target == 'gt':
                    att_reg_targets = reg_targets
                    att_labels = labels
                elif self.attack_target == 'pred':
                    att_reg_targets = ref_result['loc']
                    att_labels = ref_result['cls']
                else:
                    raise NotImplementedError(self.attack_target)

                evasion = self.attack_model(bev_seq, trans_matrices, num_agent,
                                            batch_size, anchors, reg_loss_mask, reg_targets=att_reg_targets, labels=att_labels)
                result = self.attack_model.inference(
                    bev_seq, evasion, trans_matrices, num_agent, batch_size=batch_size)

            N = bev_seq.shape[0]

            loss_collect = self.loss_calculator(
                result, anchors, reg_loss_mask, reg_targets, labels, N, motion_labels, motion_mask)
            loss_num = loss_collect[0]
            if loss_num == 3:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2 = loss_collect
            elif loss_num == 2:
                loss_num, loss, loss_cls, loss_loc = loss_collect
            elif loss_num == 4:
                loss_num, loss, loss_cls, loss_loc_1, loss_loc_2, loss_motion = loss_collect

        seq_results = [[] for i in range(NUM_AGENT)]

        for k in range(NUM_AGENT):
            bev_seq = torch.unsqueeze(data['bev_seq'][k, :, :, :, :], 0)

            if torch.nonzero(bev_seq).shape[0] == 0:
                seq_results[k] = []
            else:
                batch_box_preds = torch.unsqueeze(
                    result['loc'][k, :, :, :, :, :], 0)
                batch_cls_preds = torch.unsqueeze(result['cls'][k, :, :], 0)
                anchors = torch.unsqueeze(data['anchors'][k, :, :, :, :], 0)
                batch_motion_preds = None

                if not self.only_det:
                    if self.config.pred_type == 'center':
                        batch_box_preds[:, :, :, :, 1:,
                                        2:] = batch_box_preds[:, :, :, :, [0], 2:]

                class_selected = apply_nms_det_detectron2(
                    batch_box_preds, batch_cls_preds, anchors, self.code_type, self.config, batch_motion_preds)
                seq_results[k] = class_selected

        if validation:
            return loss.item(), loss_cls.item(), loss_loc.item(), seq_results
        else:
            return seq_results

    def median_smoothing(self, data, batch_size=1):
        # if not hasattr(self, "smooth_model"):
        #     from .smooth_model import SmoothMedianNMS, DetectionsAcc
        #     self.smooth_model = SmoothMedianNMS(self.model.module, sigma=3.6, accumulator=DetectionsAcc(loc_bin_count=3))
        
        NUM_AGENT = 5
        bev_seq = data['bev_seq']
        vis_maps = data['vis_maps']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']
        num_sensor = num_agent[0, 0]

        labels = data['labels']
        anchors = data['anchors']
        reg_targets = data['reg_targets']
        reg_loss_mask = data['reg_loss_mask']
        motion_labels = None
        motion_mask = None

        if self.attack:
            if self.attack_target == 'gt':
                att_reg_targets = reg_targets
                att_labels = labels
            elif self.attack_target == 'pred':
                if not self.attack_model.com:
                    ref_result = self.model.module.forward_no_com(
                        bev_seq, trans_matrices, num_agent, batch_size=batch_size)
                else:
                    ref_result = self.model(
                        bev_seq, trans_matrices, num_agent, batch_size=batch_size)
                att_reg_targets = ref_result['loc']
                att_labels = ref_result['cls']
            else:
                raise NotImplementedError(self.attack_target)

            evasion = self.attack_model(bev_seq, trans_matrices, num_agent,
                                        batch_size, anchors, reg_loss_mask, 
                                        reg_targets=att_reg_targets, labels=att_labels)
            attack_src, attack_tgt = self.attack_model.get_attack_pair(num_agent)
        else:
            evasion, attack_src, attack_tgt = None, None, None
            
        detections, detections_u, detections_l = self.smooth_model(n=50, 
                                                    bevs=bev_seq, trans_matrices=trans_matrices, 
                                                    num_agent_tensor=num_agent, batch_size=batch_size, 
                                                    batch_anchors=anchors, 
                                                    attack=evasion, attack_src=attack_src, attack_tgt=attack_tgt,)
        return detections

    def generate_feature_map(self, data, batch_size, merge_feature=True):
        NUM_AGENT = 5
        bev_seq = data['bev_seq']
        vis_maps = data['vis_maps']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']
        num_sensor = num_agent[0, 0]

        model = self.model.module

        features = model.encode(bev_seq)
        com_features, size = model.select_com_layer(*features)

        N = bev_seq.shape[0]

        labels = data['labels']
        anchors = data['anchors']
        reg_targets = data['reg_targets']
        reg_loss_mask = data['reg_loss_mask']


        if self.attack:
            if self.attack_model.attack_mode == 'self':
                ref_result = self.model.module.forward_no_com(
                    bev_seq, trans_matrices, num_agent, batch_size=batch_size)
            else:
                ref_result = self.model(bev_seq, trans_matrices,
                                num_agent, batch_size=batch_size)

            if self.attack_target == 'gt':
                att_reg_targets = reg_targets
                att_labels = labels
            elif self.attack_target == 'pred':
                att_reg_targets = ref_result['loc']
                att_labels = ref_result['cls']
            else:
                raise NotImplementedError(self.attack_target)

            attack = self.attack_model(bev_seq, trans_matrices, num_agent,
                                        batch_size, anchors, reg_loss_mask, reg_targets=att_reg_targets, labels=att_labels)
            attack_src, attack_tgt = self.attack_model.get_attack_pair(num_agent)

            if merge_feature:
                com_src, com_tgt = [], []
                for attack_src_, attack_tgt_ in zip(attack_src, attack_tgt):
                    com_src.append(attack_tgt_)
                    com_tgt.append(attack_tgt_)
                    
                    com_src.append(attack_src_)
                    com_tgt.append(attack_tgt_)
                com_src = torch.stack(com_src, dim=0)
                com_tgt = torch.stack(com_tgt, dim=0)
                perturb = model.place_attack_v2(
                        attack, attack_src, attack_tgt, com_src, com_tgt)
                fused_feat = model.communication_attack(
                        com_features, trans_matrices, com_src, com_tgt, size, perturb, batch_size)

                return_list = []
                for i in attack_tgt[:, 1].tolist():
                    return_list.append(
                        {"original": com_features[i].detach().cpu(),
                        "fused": fused_feat[i].detach().cpu(),}
                    )
                return return_list, \
                    attack_src.detach().cpu(), \
                    attack_tgt.detach().cpu()

            else:
                perturbed_com_features = attack + com_features[attack_src[:, 1]]
                return perturbed_com_features.detach().cpu(), \
                    attack_src.detach().cpu(), \
                    attack_tgt.detach().cpu()
        else:
            if merge_feature:
                com_srcs, com_tgts = model.get_attack_det_com_pairs(num_agent)
                com_src, com_tgt = com_srcs[-1], com_tgts[-1]
                fused_feat = model.communication_v2(
                    com_features, trans_matrices, com_src, com_tgt, size, batch_size)
                
                src, tgt = [], []
                for com_src_, com_tgt_ in zip(com_src, com_tgt):
                    if (com_src_ == com_tgt_).all():
                        continue
                    src.append(com_src_)
                    tgt.append(com_tgt_)
                src = torch.stack(src, dim=0).detach().cpu()
                tgt = torch.stack(tgt, dim=0).detach().cpu()

                return_list = []
                for i in tgt[:, 1].tolist():
                    return_list.append(
                        {"original": com_features[i].detach().cpu(),
                        "fused": fused_feat[i].detach().cpu(),}
                    )
                # import ipdb;ipdb.set_trace()
                return return_list, src, tgt
            else:
                return com_features.detach().cpu()     

