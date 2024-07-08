import os
from collections import OrderedDict

import numpy as np
import torch
import time

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error
from opencood.defense_model import RawAEDetector, ResidualAEDetector

None_dict = {"pred_box_tensor" : None, \
                "pred_score" : None, \
                "gt_box_tensor" : None,\
                "erase_index" : None,\
                "pred_and_label": None,\
                "pred_gt_box_tensor": None,
                "history": None,
                "spent_time": None}

def get_attack_save_path(attack_type, attack_conf, save_path):

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
        
    return save_path

def get_save_dir(attack_type, attack_conf, tmp_save_path = ""):

    if attack_type != 'no_attack':
        save_path = tmp_save_path
        attack_target = attack_conf.attack.attack_target
    else:
        save_path = 'no_attack/'

    if attack_type == 'pgd':
        save_path = save_path + f'{attack_target}_single_agent_{attack_conf.attack.loss_type}_loss_{attack_conf.attack.pgd.eps[0]}'
    elif attack_type == 'shift':
        save_path = save_path + f'shift_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length'
    elif attack_type == 'shift_and_pgd':
        save_path = save_path + f'shift_and_pgd_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}'
    elif attack_type == 'erase_and_shift_and_pgd':
        save_path = save_path + f'erase_and_shift_and_pgd_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}length_eps{attack_conf.attack.pgd.eps[0]}_iou{attack_conf.attack.erase.iou_thresh}'
    elif attack_type == 'erase_and_shift':
        save_path = save_path + f'erase_and_shift_{attack_target}_single_agent_{attack_conf.attack.shift.bbox_num}bbox_{attack_conf.attack.shift.shift_length}_iou{attack_conf.attack.erase.iou_thresh}'
    elif attack_type == 'no_attack':
        save_path = save_path + f'fuse_without_attack'
    
    return save_path

def normal_inference(batch_data, model, dataset, attack=None, attack_target='pred', save_path=None, num = 0, save_attack = False, attack_type = 'pgd'):
    
    tmp = batch_data.pop('history')
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    if attack_type == 'no_attack':
        save_dir = None
    else:
        from omegaconf import OmegaConf
        attack_conf = OmegaConf.load(attack)
        save_dir = get_save_dir(attack_type=attack_type, attack_conf=attack_conf, tmp_save_path=save_path)
    
    # Attack Inference
    output_dict['ego'], erase_index, spent_time = model(cav_content, attack, attack_target, save_path, num, save_attack = save_attack, attack_type = attack_type, dataset = dataset, if_single = False, save_dir = save_dir, if_att_time = True)

    # import pdb; pdb.set_trace()
    with torch.no_grad():
        pred_box_tensor, pred_score, gt_box_tensor, pred_box = \
        dataset.post_process(batch_data,
                                output_dict)
        
    pred_gt_box_tensor = None
    # erase_index = None
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor,\
                    "erase_index" : erase_index,\
                    "pred_and_label": None,\
                    "pred_gt_box_tensor": pred_gt_box_tensor,
                    'spent_time': spent_time}
    
    return return_dict

def get_attack(attack_type, attack_conf, save_path, num = 0):
    
    if attack_type == 'no_attack':
        attack = {'src': []}
    else:
        tmp_save_path = get_attack_save_path(attack_type=attack_type, attack_conf=attack_conf, save_path = save_path)
        if os.path.exists(tmp_save_path + f'sample_{num}.npy'):
            attack = np.load(tmp_save_path + f'sample_{num}.npy',allow_pickle=True)
        else:
            return_dict = None_dict
            return True, return_dict
    
    return False, attack

def attack_detection(data_dict, dataset, method, attack = None, num = 0, save_dir = 'test_set_pred', attack_type = 'pgd', attack_conf = None, defense_model = None, robosac_cfg = None, model = None, match_para = 1, multi_test_alpha = 0.05):
    
    if method != 'robosac':
        
        tmp = data_dict.pop('history')
    
    if method == 'match':
        from opencood.models.defense_model.attack_detection_match import AttackDetectionMatch
        detector = AttackDetectionMatch(model)
        time1 = time.time()
        pred_box_tensor, pred_score, gt_box_tensor, pred_and_label = detector.forward(data_dict, dataset, attack, num = num, save_dir=save_dir, attack_type = attack_type, attack_conf = attack_conf, match_para = match_para)
        history = None
        return pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history, time.time() - time1
    elif method == 'residual' or method == 'raw':
        from opencood.models.defense_model.attack_detection_ae import AttackDetectionAe
        detector = AttackDetectionAe(model)
        time1 = time.time()
        pred_box_tensor, pred_score, gt_box_tensor, pred_and_label = detector.forward(data_dict, dataset, attack, num = num, save_dir = save_dir, attack_type = attack_type,  method=method, attack_conf=attack_conf, defense_model = defense_model)
        history = None
        return pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history, time.time() - time1
    elif method == 'multi_test':
        from opencood.models.defense_model.attack_detection_multi_test import AttackDetectionMultiTest
        detector = AttackDetectionMultiTest(model)
        time1 = time.time()
        pred_box_tensor, pred_score, gt_box_tensor, pred_and_label = detector.forward(data_dict, dataset, attack, num = num, attack_type=attack_type, attack_conf=attack_conf, save_dir = save_dir, multi_test_alpha = multi_test_alpha)
        history = None
        return pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history, time.time() - time1
    elif method == 'robosac':
        from opencood.models.defense_model.robosac import ROBOSAC
        if robosac_cfg is None:
            detector = ROBOSAC(model)
        else:
            from omegaconf import OmegaConf
            robosac_conf = OmegaConf.load(robosac_cfg)
            detector = ROBOSAC(model, **robosac_conf)
        time1 = time.time()
        pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history = detector.forward(data_dict, dataset, attack_conf, num = num, attack = attack, save_dir = save_dir, attack_type = attack_type)
        return pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history, time.time() - time1
    else:
        print("Please choose an available method among match,autoencoder and multi_test!")
        exit(0)

def inference_early_fusion(batch_data, model, dataset, attack=None, attack_target='pred', attack_conf = None, save_path=None, num = 0, save_attack = False, standard = None, attack_type = 'pgd', ae_type = None, defense_model = None, temperature = 20, robosac_cfg = None, match_para = 1, multi_test_alpha = 0.05):
    
    if standard == None:
        return_dict = normal_inference(batch_data, model, dataset, attack, attack_target, save_path, num, save_attack, attack_type)
        
    elif standard.startswith('g_match_'):

        flag, attack = get_attack(attack_type, attack_conf, save_path, num)
        if flag:
            return attack
        save_dir = standard[8:]
        from opencood.models.generate_sample.generate_match import generate_match
        generator = generate_match(model)
        tmp = batch_data.pop('history')
        generator.forward(batch_data, dataset, attack = attack, num = num, save_dir = save_dir, match_para = match_para)
        return_dict = None_dict
    
    # For Ae in Train set
    elif standard.startswith('g_ae_train_'):
        
        flag, attack = get_attack(attack_type, attack_conf, save_path, num)
        if flag:
            return attack
        save_dir = standard[11:]
        from opencood.models.generate_sample.generate_ae_train import generate_ae_train
        generator = generate_ae_train(model)
        generator.forward(batch_data, num = num, save_dir = save_dir, ae_type = ae_type, attack = attack, attack_type = attack_type, attack_conf = attack_conf, dataset = dataset)
        return_dict = None_dict
    
    # For Ae in Validation set
    elif standard.startswith('g_ae_val_'):

        flag, attack = get_attack(attack_type, attack_conf, save_path, num)
        if flag:
            return attack
        save_dir = standard[9:]
        from opencood.models.generate_sample.generate_ae_val import generate_ae_val
        generator = generate_ae_val(model)
        generator.forward(batch_data, num = num, save_dir = save_dir, ae_type = ae_type, attack = attack, attack_type = attack_type, defense_model = defense_model, dataset = dataset, attack_conf = attack_conf)
        return_dict = None_dict
        
    else:
        # Run different defense methods
        flag, attack = get_attack(attack_type, attack_conf, save_path, num)
        if flag:
            return attack
        
        save_dir = get_save_dir(attack_type=attack_type, attack_conf=attack_conf, tmp_save_path=save_path)
        history = None

        if standard == 'match':
            pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history, spent_time = attack_detection(batch_data, dataset, attack = attack, method = 'match', num = num, save_dir = save_dir, attack_type = attack_type, attack_conf = attack_conf, model = model, match_para = match_para)
        
        if standard == 'multi_test':
            pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history, spent_time = attack_detection(batch_data, dataset, attack = attack, method = 'multi_test', num = num, save_dir = save_dir, attack_type = attack_type, attack_conf = attack_conf, model = model, multi_test_alpha = multi_test_alpha)

        if standard == 'ae':
            pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history, spent_time = attack_detection(batch_data, dataset, attack = attack, method = ae_type, num = num, save_dir = save_dir, attack_type = attack_type, attack_conf = attack_conf, defense_model = defense_model, model = model)
            
        if standard == 'robosac':
            pred_box_tensor, pred_score, gt_box_tensor, pred_and_label, history, spent_time = attack_detection(batch_data, dataset, attack = attack, method = 'robosac', num = num, save_dir = save_dir, attack_type = attack_type, attack_conf = attack_conf, robosac_cfg=robosac_cfg, model = model)

        return_dict = {"pred_box_tensor" : pred_box_tensor, \
                        "pred_score" : pred_score, \
                        "gt_box_tensor" : gt_box_tensor,\
                        "erase_index" : None,\
                        "pred_gt_box_tensor": None,\
                        "pred_and_label": pred_and_label,
                        "history": history,
                        "spent_time": spent_time}

    #########################################################

    return return_dict

def inference_intermediate_fusion(batch_data, model, dataset, attack=None, attack_target='pred', attack_conf = None, save_path=None, num = 0, save_attack = False, standard = None, attack_type = 'pgd', ae_type = None, defense_model = None, temperature = 20, robosac_cfg = None, match_para = 1, multi_test_alpha = 0.05):

    return_dict = inference_early_fusion(batch_data, model, dataset, attack, attack_target, attack_conf, save_path, num, save_attack, standard, attack_type, ae_type, defense_model, temperature, robosac_cfg, match_para, multi_test_alpha)
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