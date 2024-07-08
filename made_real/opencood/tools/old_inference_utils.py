# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error
from opencood.defense_model import RawAEDetector, ResidualAEDetector

def get_attack_path(attack_type, attack_conf, save_path):

    # save_path = '/GPFS/data/shengyin/OpenCOOD-main/save_attack/'
    attack_target = attack_conf.attack.attack_target

    if attack_type == 'pgd':
        save_path = save_path + f'_{attack_conf.attack.pgd.eps[0]}/'
    elif attack_type == 'shift':
        save_path = save_path + f'_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length/'
    elif attack_type == 'shift_and_pgd':
        save_path = save_path + f'_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}/'
    elif attack_type == 'erase_and_shift_and_pgd':
        save_path = save_path + f'_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}_iou{attack_conf.attack.erase.iou_thresh}/'
    elif attack_type == 'erase_and_shift':
        save_path = save_path + f'_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_iou{attack_conf.attack.erase.iou_thresh}/'
    elif attack_type == 'shift_and_pgd_fg':
        save_path = save_path + f'_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}/'
    elif attack_type == 'rotate_and_pgd':
        save_path = save_path + f'_{attack_conf.attack.rotate.bbox_num}bbox_{attack_conf.attack.rotate.shift_angle}angle_eps{attack_conf.attack.pgd.eps[0]}/'
    elif attack_type == 'rotate':
        save_path = save_path + f'_{attack_conf.attack.rotate.bbox_num}bbox_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}/'
    elif attack_type == 'shift_and_rotate':
        save_path = save_path + f'_{attack_conf.attack.bbox_num}bbox_shift_{attack_conf.attack.shift.shift_length}_{attack_conf.attack.shift.shift_direction}_rotate_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}/'
        
    return save_path

def get_save_dir(attack_type, attack_conf, tmp_save_path = None):

    if attack_type != 'no attack':

        save_path = 'test_set_' + tmp_save_path
        attack_target = attack_conf.attack.attack_target

        if tmp_save_path == 'o':
            save_path = save_path + 'out_range_'
    else:
        save_path = 'test_set_single_agent/'

    if attack_type == 'pgd':
        save_path = save_path + f'{attack_target}_single_agent_{attack_conf.attack.loss_type}_loss_{attack_conf.attack.pgd.eps[0]}'
    elif attack_type == 'shift':
        save_path = save_path + f'shift_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length'
    elif attack_type == 'shift_and_pgd':
        save_path = save_path + f'shift_and_pgd_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}'
    elif attack_type == 'shift_and_pgd_fg':
        save_path = save_path + f'shift_and_pgd_fg_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}'
    elif attack_type == 'rotate_and_pgd':
        save_path = save_path + f'rotate_and_pgd_{attack_target}_single_agent_{attack_conf.attack.rotate.bbox_num}bbox_{attack_conf.attack.rotate.shift_angle}angle_eps{attack_conf.attack.pgd.eps[0]}'
    elif attack_type == 'rotate':
        save_path = save_path + f'rotate_{attack_target}_single_agent_{attack_conf.attack.rotate.bbox_num}bbox_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}'
    elif attack_type == 'shift_and_rotate':
        save_path = save_path + f'shift_and_rotate_{attack_target}_single_agent_{attack_conf.attack.bbox_num}bbox_shift_{attack_conf.attack.shift.shift_length}_{attack_conf.attack.shift.shift_direction}_rotate_{attack_conf.attack.rotate.shift_type}_{attack_conf.attack.rotate.shift_angle}'
    elif attack_type == 'erase_and_shift_and_pgd':
        save_path = save_path + f'erase_and_shift_and_pgd_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}_iou{attack_conf.attack.erase.iou_thresh}'
    elif attack_type == 'erase_and_shift':
        save_path = save_path + f'erase_and_shift_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}_iou{attack_conf.attack.erase.iou_thresh}'
    elif attack_type == 'no attack':
        save_path = save_path + f'fuse_without_attack'
    
    return save_path

def inference_early_fusion(batch_data, model, dataset, attack=False, com=True, attack_mode='self', eps=0.1, alpha=0.1, proj=True, attack_target='pred', save_path=None, step=15, noise_attack=False, num = 0, save_attack = False, standard = None, attack_type = 'pgd', ae_type = None, defense_model = None):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """

    if standard == None:

        output_dict = OrderedDict()
        output_dict1 = OrderedDict()
        cav_content = batch_data['ego']
        record_len = cav_content['record_len']
        if attack_type == 'None':
            save_dir = None
        else:
            from omegaconf import OmegaConf
            attack_conf = OmegaConf.load(attack)
            save_dir = get_save_dir(attack_type=attack_type, attack_conf=attack_conf, tmp_save_path=save_path)
        
        # Attack Inference
        output_dict['ego'], erase_index = model(cav_content, attack, attack_target, save_path, num, save_attack = save_attack, attack_type = attack_type, dataset = dataset, if_single = False, save_dir = save_dir)

        with torch.no_grad():
            pred_box_tensor, pred_score, gt_box_tensor, pred_box = \
            dataset.post_process(batch_data,
                                 output_dict)
        
        # 保存confidence和regression的结果
        # path = 'outcome/confidence_and_regression' + '/temp20_[1,2]/'
        # if os.path.exists(path) == False:
        #     os.makedirs(path)
        # np.save(path + f'pred_box_{num}.npy', pred_box_tensor.cpu().numpy())
        # np.save(path + f'pred_score_{num}.npy', pred_score.cpu().numpy())
            
        pred_gt_box_tensor = None
        # erase_index = None
        
        return_dict = {"pred_box_tensor" : pred_box_tensor, \
                        "pred_score" : pred_score, \
                        "gt_box_tensor" : gt_box_tensor,\
                        "erase_index" : erase_index,\
                        "pred_and_label": None,\
                        "pred_gt_box_tensor": pred_gt_box_tensor}

    elif standard.startswith('g_match_'):

        # Generate match cost / ae loss
        attack = {'src': []}
        save_dir = standard[8:]
        model.generate_match_loss(batch_data, dataset, attack = attack, num = num, save_dir = save_dir)
        return_dict = {"pred_box_tensor" : None, \
                        "pred_score" : None, \
                        "gt_box_tensor" : None,\
                        "erase_index" : None,\
                        "pred_and_label": None,\
                        "pred_gt_box_tensor": None}
    
    # For Ae in Train set
    elif standard.startswith('g_ae_train_'):
        # attack = {'src': []}

        from omegaconf import OmegaConf
        attack_conf = OmegaConf.load(attack)
        if attack_type == 'no attack':
            attack = {'src': []}
        else:
            tmp_save_path = get_attack_path(attack_type=attack_type, attack_conf=attack_conf, save_path = save_path)
            if os.path.exists(tmp_save_path + f'sample_{num}.npy'):
                attack = np.load(tmp_save_path + f'sample_{num}.npy',allow_pickle=True)
            else:
                return_dict = {"pred_box_tensor" : None, \
                        "pred_score" : None, \
                        "gt_box_tensor" : None,\
                        "erase_index" : None,\
                        "pred_and_label": None,\
                        "pred_gt_box_tensor": None}
                return return_dict
                

        save_dir = standard[11:]
        model.generate_ae_train(batch_data, num = num, save_dir = save_dir, ae_type = ae_type, attack = attack, attack_type = attack_type, attack_conf = attack_conf, dataset = dataset)
        return_dict = {"pred_box_tensor" : None, \
                        "pred_score" : None, \
                        "gt_box_tensor" : None,\
                        "erase_index" : None,\
                        "pred_and_label": None,\
                        "pred_gt_box_tensor": None}
    
    # For Ae in Validation set
    elif standard.startswith('g_ae_val_'):

        # Generate residual ae loss
        attack = {'src': []}
        save_dir = standard[9:]
        model.generate_ae_val(batch_data, num = num, save_dir = save_dir, ae_type = ae_type,  defense_model = defense_model)
        return_dict = {"pred_box_tensor" : None, \
                        "pred_score" : None, \
                        "gt_box_tensor" : None,\
                        "erase_index" : None,\
                        "pred_and_label": None,\
                        "pred_gt_box_tensor": None}
        
    else:
        # Run different defense methods
        from omegaconf import OmegaConf
        attack_conf = OmegaConf.load(attack)
        if attack_type == 'no attack':
            attack = {'src': []}
        else:
            tmp_save_path = get_attack_path(attack_type=attack_type, attack_conf=attack_conf, save_path = save_path)
            attack = np.load(tmp_save_path + f'sample_{num}.npy',allow_pickle=True)
        
        save_dir = get_save_dir(attack_type=attack_type, attack_conf=attack_conf, tmp_save_path=save_path)

        if standard == 'match':
            save_dir = save_dir
            pred_box_tensor, pred_score, gt_box_tensor, pred_and_label = model.attack_detection(batch_data, dataset, attack = attack, method = 'match', num = num, save_dir = save_dir, attack_type = attack_type, attack_conf = attack_conf)
        
        if standard == 'multi_test':
            pred_box_tensor, pred_score, gt_box_tensor, pred_and_label = model.attack_detection(batch_data, dataset, attack = attack, method = 'multi_test', num = num, save_dir = save_dir, attack_type = attack_type, attack_conf = attack_conf)

        if standard == 'ae':
            pred_box_tensor, pred_score, gt_box_tensor, pred_and_label = model.attack_detection(batch_data, dataset, attack = attack, method = ae_type, num = num, save_dir = save_dir, attack_type = attack_type, attack_conf = attack_conf, defense_model = defense_model)

        return_dict = {"pred_box_tensor" : pred_box_tensor, \
                        "pred_score" : pred_score, \
                        "gt_box_tensor" : gt_box_tensor,\
                        "erase_index" : None,\
                        "pred_gt_box_tensor": None,\
                        "pred_and_label": pred_and_label}

    #########################################################

    # Unknown
    # if "depth_items" in output_dict['ego']:
    #     return_dict.update({"depth_items" : output_dict['ego']['depth_items']})

    return return_dict

def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict


def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {'ego': batch_data['ego']}
        
    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
        dataset.post_process_no_fusion_uncertainty(batch_data, # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "uncertainty_tensor" : uncertainty_tensor}

    return return_dict


def inference_intermediate_fusion(batch_data, model, dataset, attack=False, com=True, attack_mode='self', eps=0.1, alpha=0.1, proj=True, attack_target='pred', save_path=None, step=15, noise_attack=False, num = 0, save_attack = False, standard = None, attack_type = 'pgd', ae_type = None, defense_model = None):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return_dict = inference_early_fusion(batch_data, model, dataset, attack, com, attack_mode, eps, alpha, proj, attack_target, save_path, step, noise_attack, num=num, save_attack=save_attack, standard = standard, attack_type = attack_type, ae_type=ae_type, defense_model = defense_model)
    return return_dict

def inference_intermediate_fusion_multiclass(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_multiclass(batch_data,
                             output_dict)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    return return_dict

def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def depth_metric(depth_items, grid_conf):
    # depth logdit: [N, D, H, W]
    # depth gt indices: [N, H, W]
    depth_logit, depth_gt_indices = depth_items
    depth_pred_indices = torch.argmax(depth_logit, 1)
    depth_pred = indices_to_depth(depth_pred_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    depth_gt = indices_to_depth(depth_gt_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    rmse = mean_squared_error(depth_gt.cpu(), depth_pred.cpu(), squared=False)
    return rmse


def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data):
    """
    Fix the missing pred_box and gt_box for ego and cav(s).
    Args:
        pred_box_tensor : tensor
            shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
        gt_box_tensor : tensor
            shape (N2, 8, 3), not include ego agent in camera cases, but it should include
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    Returns:
        pred_box_tensor : tensor
            shape (N1+?, 8, 3)
        gt_box_tensor : tensor
            shape (N2+1, 8, 3)
    """
    if pred_box_tensor is None or gt_box_tensor is None:
        return pred_box_tensor, gt_box_tensor, pred_score, 0
    # prepare cav's boxes

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()
        
    extent = [2.45, 1.06, 0.75]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)
    
    pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
    gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

    pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

    gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
    gt_score_[N:] = 0.5

    keep_index = nms_rotated(pred_box_tensor_,
                            pred_score_,
                            0.01)
    pred_box_tensor = pred_box_tensor_[keep_index]
    pred_score = pred_score_[keep_index]

    keep_index = nms_rotated(gt_box_tensor_,
                            gt_score_,
                            0.01)
    gt_box_tensor = gt_box_tensor_[keep_index]

    return pred_box_tensor, gt_box_tensor, pred_score, N


def get_cav_box(batch_data):
    """
    Args:
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    """

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
        agent_modality_list = batch_data['ego']['agent_modality_list']

    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        agent_modality_list = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
            agent_modality_list.append(cav_data['modality_name'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

        

    extent = [0.2, 0.2, 0.2]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_np = np.concatenate(box_list, axis=0)


    return cav_box_np, agent_modality_list