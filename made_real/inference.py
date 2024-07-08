import argparse
import os
import time
from typing import OrderedDict
import importlib       
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import inference_utils, train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils      
from opencood.visualization import vis_utils, my_vis, simple_vis
torch.multiprocessing.set_sharing_strategy('file_system')

import datetime
from opencood.utils.generate_npy import generate_npy
from opencood.utils.generate_pred import generate_pred
from opencood.utils.precision_and_recall import compute_precision_and_recall
from tqdm import tqdm
from opencood.defense_model import RawAEDetector, ResidualAEDetector
from config.defense.percentiles import *


def test_parser():

    # inference args
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--modal', type=int, default=0,
                        help='used in heterogeneous setting, 0 lidaronly, 1 camonly, 2 ego_lidar_other_cam, 3 ego_cam_other_lidar')
    parser.add_argument('--note', default="", type=str, help="any other thing?")

    # add args about visualization
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--vis_no_score', action='store_true',
                        help="whether print the score of prediction")
    
    # add args about datasets
    parser.add_argument('--dataset', type=str, default='test', help="use test set or validation set")
    parser.add_argument('--range_dataset', type=str, default='[]', help="use part of dataset")
    parser.add_argument('--temperature', type=int, default=20, help="temperature for fusion")

    # add attack args
    parser.add_argument('--attack', type=str, default=None, help="Attack config file, "
                            "if it is \"TRUE\", following attack hyperparameters will be used")

    # add defense methods
    parser.add_argument('--standard', type=str, default=None, help='choose a standard')
    parser.add_argument('--ae_type', type=str, default=None, help='choose a standard')
    parser.add_argument('--robosac_cfg', type=str, default=None, help="robosac config file")
    parser.add_argument('--match_para', type=float, default=1, help="parameter for match")
    parser.add_argument('--multi_test_alpha', type=float, default=0.05, help="control FPR for multi test")

    opt = parser.parse_args()
    return opt

def seed_it(seed):
    # random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = True  
    torch.manual_seed(seed)

# create the dictionary for evaluation
def create_result_stat(num = 5):
    result_stat = []
    for i in range(num):
        tmp_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
        }
        result_stat.append(tmp_stat)
    return result_stat

def init_pre():
    
    seed_it(2159)
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 
    hypes = yaml_utils.load_yaml(None, opt)

    # Unknown
    if 'heter' in hypes:
        if opt.modal == 0:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += '_lidaronly' 

        if opt.modal == 1:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_camonly' 
            
        if opt.modal == 2:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += 'ego_lidar_other_cam'

        if opt.modal == 3:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_ego_cam_other_lidar'

        x_min, x_max = -140.8, 140.8
        y_min, y_max = -40, 40
        opt.note += f"_{x_max}_{y_max}"
        hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]]
        hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]]
        hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        
        hypes['preprocess']['cav_lidar_range'] =  new_cav_range
        hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['gt_range'] = new_cav_range
        hypes['model']['args']['lidar_args']['lidar_range'] = new_cav_range
        if 'camera_mask_args' in hypes['model']['args']:
            hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = new_cav_range

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)
        
    return opt, hypes

def create_dataset_and_model(opt, hypes):
    
    # Dataset Address
    if opt.dataset == 'test':
        hypes['validate_dir'] = hypes['test_dir']
        print("Processing Test Set!!!!!!!!!!!!!!!!")
    else:
        hypes['validate_dir'] = hypes['root_dir']
        print("Processing Train Set!!!!!!!!!!!!!!!!")
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes, temperature=opt.temperature)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    # build dataset for each noise setting
    print('Dataset Building')
    if opt.dataset == 'train':
        opencood_dataset = build_dataset(hypes, visualize=True, train=True)
    else:
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)

    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    validation_index = np.load('validation_index.npy')
    train_index = np.load('train_index.npy')
    # test_index = np.load('new_test_index.npy')

    # For Attack to Inference
    range_of_data = eval(opt.range_dataset)
    if opt.dataset == 'validation':
        range_of_data = validation_index
    elif opt.dataset == 'train':
        range_of_data = train_index
        
    # import ipdb; ipdb.set_trace()
    if range_of_data == [] and opt.dataset == 'test':
        data_loader = DataLoader(opencood_dataset,
                        batch_size=1,
                        num_workers=4,
                        collate_fn=opencood_dataset.collate_batch_test,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)
    elif range_of_data != [] and opt.dataset == 'test':
        range_of_data = range(range_of_data[0],range_of_data[1])
        data_loader = DataLoader(Subset(opencood_dataset, range_of_data),
                        batch_size=1,
                        num_workers=4,
                        collate_fn=opencood_dataset.collate_batch_test,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)
    else:
        data_loader = DataLoader(Subset(opencood_dataset, range_of_data),
                        batch_size=1,
                        num_workers=4,
                        collate_fn=opencood_dataset.collate_batch_test,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)

    print(f"There are {len(data_loader)} samples!!!")
    
    return device, model, data_loader, opencood_dataset, left_hand

def load_attack_conf(attack):
    
    if attack == None:
        return None, 'pred', 'no_attack', 'no_loss', None, False, None
    
    from omegaconf import OmegaConf
    attack_conf = OmegaConf.load(attack)
    attack_type = attack_conf.attack.attack_type
    attack_target = attack_conf.attack.attack_target
    loss_type = attack_conf.attack.loss_type
    save_path = attack_conf.attack.save_path + f'_{attack_target}'
    save_attack = attack_conf.attack.save_attack
    
    remaining_dict = {}
    if 'pgd' in attack_type:
        eps = attack_conf.attack.pgd.eps[0]
        alpha = attack_conf.attack.pgd.alpha[0]
        steps = attack_conf.attack.pgd.steps
        
        remaining_dict.update({'eps': eps, 'alpha': alpha, 'steps': steps})
    
    if 'shift' in attack_type:
        bbox_num = attack_conf.attack.shift.bbox_num
        shift_length = attack_conf.attack.shift.shift_length
        padding_type = attack_conf.attack.shift.padding_type
        remaining_dict.update({'bbox_num': bbox_num, 'shift_length': shift_length, 'padding_type': padding_type})
        
    if 'erase' in attack_type:
        iou = attack_conf.attack.erase.iou_thresh
        remaining_dict.update({'iou': iou})
    
    return attack_conf, attack_target, attack_type, loss_type, save_path, save_attack, remaining_dict

def get_defense_model(opt):
    
    if opt.ae_type == 'residual':
        # defense_model = None
        defense_model = ResidualAEDetector(384, ckpt="opencood/ae_train/1016-res-ae/autoencoder_res_199.pth", threshold = residual_ae_percentiles[95]).cuda()
        # defense_model = ResidualAEDetector(64, ckpt="opencood/ae_train/1010-AE-train/layer0/Layer_0_autoencoder_res_199.pth", threshold = residual_ae_percentiles[95]).cuda()
        # defense_model = RawAEDetector(128, ckpt="opencood/ae_train/1016-RAW-AE-train/layer1/Layer_1_autoencoder_raw_199.pth", threshold = raw_ae_percentiles[1], layer_num=1).cuda()
        
    elif opt.ae_type == 'raw':
        defense_model_0 = RawAEDetector(64, ckpt="opencood/ae_train/1016-RAW-AE-train/layer0/Layer_0_autoencoder_raw_199.pth", threshold = raw_ae_percentiles[0], layer_num=0).cuda()
        defense_model_1 = RawAEDetector(128, ckpt="opencood/ae_train/1016-RAW-AE-train/layer1/Layer_1_autoencoder_raw_199.pth", threshold = raw_ae_percentiles[1], layer_num=1).cuda()
        defense_model_2 = RawAEDetector(256, ckpt="opencood/ae_train/1007-raw-train/layer2/Layer_2_autoencoder_raw_199.pth", threshold = raw_ae_percentiles[2], layer_num=2).cuda()
        defense_model = [defense_model_0, defense_model_1, defense_model_2]
    else:
        defense_model = None
        
    return defense_model
    
def visualize_result(pred_box_tensor, pred_score, batch_data, infer_result, opt, hypes, left_hand, sample_idx, attack_conf, time_str, attack_target, attack_type, loss_type, pred_gt_box_tensor, infer_info, remaining_dict = None, erase_index = None):
    
    if 'pgd' in attack_type:
        eps = remaining_dict['eps']
        alpha = remaining_dict['alpha']
        steps = remaining_dict['steps']
        
    if 'shift' in attack_type:
        bbox_num = remaining_dict['bbox_num']
        shift_length = remaining_dict['shift_length']
        padding_type = remaining_dict['padding_type']
    
    if 'erase' in attack_type:
        iou = remaining_dict['iou']
    
    if (pred_box_tensor is not None):
                    
        if opt.attack_type == 'pgd':
            save_dir = f'pgd_eps{eps}_{attack_target}_{loss_type}'
        elif opt.attack_type == 'shift':
            save_dir = f'{time_str}_shift_{bbox_num}bbox_{shift_length}length_{attack_target}_padding_{attack_conf.attack.shift.padding_type}'
        elif opt.attack_type == 'shift_and_pgd':
            save_dir = f'{time_str}_shift_and_pgd_{bbox_num}bbox_{shift_length}length_eps{eps}_{attack_target}'
        elif opt.attack_type == 'erase_and_shift_and_pgd':
            save_dir = f'{time_str}_erase_and_shift_and_pgd_{bbox_num}bbox_{shift_length}length_eps{eps}_{attack_target}_iou{iou}'
        elif opt.attack_type == 'erase_and_shift':
            save_dir = f'{time_str}_erase_and_shift_{bbox_num}bbox_{shift_length}length_{attack_target}_iou{iou}_padding_{attack_conf.attack.shift.padding_type}'
        else:
            save_dir = f'{time_str}_without_attack'

        if opt.attack_type != 'None':
            save_dir = save_dir + attack_conf.attack.loss_type
        if opt.standard != None:
            save_dir = opt.standard + '_' + save_dir

        # import ipdb; ipdb.set_trace()
        vis_save_path_root = os.path.join(opt.model_dir, save_dir, f'vis_{infer_info}')
        if not os.path.exists(vis_save_path_root):
            os.makedirs(vis_save_path_root)
        
        vis_save_path = os.path.join(vis_save_path_root, f'sample_{sample_idx}.png')
        simple_vis.visualize(infer_result,
                            batch_data['ego'][
                                'origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            left_hand=left_hand,
                            confidence=pred_score, erase_index = erase_index, pred_gt_box_tensor = pred_gt_box_tensor)

def data_post_process(pred_box_tensor, pred_score, gt_box_tensor, result_stat, opt, idx, infer_result, tp_list, fp_list, num_agent, batch_data, opencood_dataset):
    
    # caluclate tp and fp
    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                pred_score,
                                gt_box_tensor,
                                result_stat[0],
                                0.3, tp_list, fp_list)
    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                pred_score,
                                gt_box_tensor,
                                result_stat[num_agent],
                                0.3, tp_list, fp_list, num_agent=num_agent)
    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                pred_score,
                                gt_box_tensor,
                                result_stat[0],
                                0.5, tp_list, fp_list)
    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                pred_score,
                                gt_box_tensor,
                                result_stat[num_agent],
                                0.5, tp_list, fp_list, num_agent=num_agent)
    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                pred_score,
                                gt_box_tensor,
                                result_stat[0],
                                0.7, tp_list, fp_list)
    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                pred_score,
                                gt_box_tensor,
                                result_stat[num_agent],
                                0.7, tp_list, fp_list, num_agent=num_agent)

    if opt.save_npy:
        npy_save_path = os.path.join(opt.model_dir, 'npy')
        if not os.path.exists(npy_save_path):
            os.makedirs(npy_save_path)
        inference_utils.save_prediction_gt(pred_box_tensor,
                                        gt_box_tensor,
                                        batch_data['ego'][
                                            'origin_lidar'][0],
                                        idx,
                                        npy_save_path)
    if getattr(opencood_dataset, "heterogeneous", False):
        cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
        infer_result.update({"cav_box_np": cav_box_np, \
                                "lidar_agent_record": lidar_agent_record})

def main():
    
    opt, hypes = init_pre()
    device, model, data_loader, opencood_dataset, left_hand = create_dataset_and_model(opt, hypes)
    attack_conf, attack_target, attack_type, loss_type, save_path, save_attack, remaining_dict = load_attack_conf(opt.attack)
    defense_model = get_defense_model(opt)
    
    # Create the dictionary for evaluation
    result_stat = create_result_stat(num = 5)
    infer_info = opt.fusion_method + opt.note
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%m-%d %H:%M')
    tp_list = {0.3:[], 0.5:[], 0.7:[]}
    fp_list = {0.3:[], 0.5:[], 0.7:[]}
    pred_and_label_list = []
    standard_spent_time_list = []
    all_spent_time_list = []
    history = None
    
    # Inference 
    for idx, batch_data in tqdm(enumerate(data_loader)):
        
        if batch_data is None:
            continue

        batch_data = train_utils.to_device(batch_data, device)
        num_agent = batch_data['ego']['record_len'][0]
        sample_idx = batch_data['ego']['sample_idx']
        batch_data['history'] = history

        if opt.fusion_method == 'late':
            infer_result = inference_utils.inference_late_fusion(batch_data,
                                                    model,
                                                    opencood_dataset)
        elif opt.fusion_method == 'early':
            infer_result = inference_utils.inference_early_fusion(batch_data,
                                                    model,
                                                    opencood_dataset)
        elif opt.fusion_method == 'intermediate':
            time1 = time.time()
            infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                    model, opencood_dataset, attack=opt.attack, attack_target=attack_target, \
                    save_path=save_path, num = sample_idx, save_attack = save_attack, attack_conf = attack_conf, \
                    standard = opt.standard, attack_type = attack_type, ae_type = opt.ae_type, defense_model = defense_model, temperature = opt.temperature, robosac_cfg = opt.robosac_cfg, match_para = opt.match_para, multi_test_alpha = opt.multi_test_alpha)
            if num_agent == 2:
                all_spent_time_list.append(time.time() - time1)
        else:
            raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                    'fusion is supported.')

        # get detailed results
        pred_box_tensor = infer_result['pred_box_tensor']
        gt_box_tensor = infer_result['gt_box_tensor']
        pred_score = infer_result['pred_score']
        erase_index = infer_result['erase_index']
        pred_gt_box_tensor = infer_result['pred_gt_box_tensor']
        pred_and_label = infer_result['pred_and_label']
        spent_time = infer_result['spent_time']
        if num_agent == 2:
            if spent_time == None:
                spent_time = 0
            standard_spent_time_list.append(spent_time)
        if pred_and_label != [] and pred_and_label != None:
            pred_and_label_list.append(pred_and_label)
        if 'history' in infer_result:
            history = infer_result['history']
        
        if opt.standard == None or not opt.standard.startswith('g_'):
            
            # data post-process
            data_post_process(pred_box_tensor, pred_score, gt_box_tensor, result_stat, opt, idx, infer_result, tp_list, fp_list, num_agent, batch_data, opencood_dataset)

            # save picture every a certain intervals
            if opt.show_vis or opt.save_vis:
                visualize_result(pred_box_tensor, pred_score, batch_data, infer_result, opt, hypes, left_hand, sample_idx, attack_conf, time_str, attack_target, attack_type, loss_type, pred_gt_box_tensor, infer_info, remaining_dict, erase_index)
                
            torch.cuda.empty_cache()


    # Print Final AP Result
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%m-%d %H:%M')
    all_time_mean = np.array(all_spent_time_list).mean()
    standard_time_mean = np.array(standard_spent_time_list).mean()
    if opt.standard != None and not opt.standard.startswith('g_'):
        
        if not os.path.exists(f'outcome/performance/{opt.standard}/'):
            os.makedirs(f'outcome/performance/{opt.standard}/')

        file_handle = open(f'outcome/performance/{opt.standard}/'+ time_str + '.txt', mode='a')
        file_handle.write(f"{attack_conf}\n")
        file_handle.write(f"all_time_mean: {all_time_mean}\n")
        file_handle.write(f"standard_time_mean: {standard_time_mean}\n")
        file_handle.write(f"{attack_conf}\n")
        for j in range(5):
            eval_utils.eval_final_results(result_stat[j],
                                    opt.model_dir, j, file_handle)
    else:
        if not os.path.exists('outcome/performance/attack/'):
            os.makedirs('outcome/performance/attack/')
        file_handle = open(f'outcome/performance/attack/'+ time_str + '.txt', mode='a')
        file_handle.write(f"standard_time_mean: {standard_time_mean}\n")
        if opt.attack != False:
            file_handle.write(f"{attack_conf}\n")
        for j in range(5):
            eval_utils.eval_final_results(result_stat[j],
                                    opt.model_dir, j, file_handle)
    
    # Compute Preision and Recall
    if opt.standard != None and not opt.standard.startswith('g_'):
        precision, recall, fpr, tp, tn, fp, fn = compute_precision_and_recall(pred_and_label_list)
        if not os.path.exists(f'outcome/pred_and_label/'):
            os.makedirs(f'outcome/pred_and_label/')
        np.save('outcome/pred_and_label' + '/' + time_str  + f'{opt.standard}.npy', pred_and_label_list)
        print(f"The precision is {precision}")
        print(f'The recall is {recall}')
        print(f'The false positive rate is {fpr}')

        if not os.path.exists(f'outcome/precision_and_recall/{opt.standard}/'):
            os.makedirs(f'outcome/precision_and_recall/{opt.standard}/')
        file_handle = open(f'outcome/precision_and_recall/{opt.standard}/'+ time_str + '.txt', mode='a')
        file_handle.write(f"{attack_conf}\n")
        file_handle.write(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}\n")
        file_handle.write(f'The precision is {precision} and the recall is {recall} and the fpr is {fpr}.\n')

if __name__ == '__main__':
    main()
