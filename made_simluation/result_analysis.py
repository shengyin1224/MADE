""""
      Analyze the results of the 1-agent communication
    and the results of matching algorithm.

    setting: 
        攻击在多agent通信的情况下产生，在两个agent通信的情况下分析攻击
        （需要对攻击的迁移性进行分析，n agent -> 2 agent）
"""

from typing import List, Dict
import os 
import glob 
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import metrics
import torch
import torch.nn.functional as F
import torch.utils.data as data
from data.config import Config, ConfigGlobal
from data.Dataset import V2XSIMDataset
from utils.mean_average_precision import EvalWorker
from utils.utils import get_pred_box

###################
# global variables
###################
config = Config('test', binary=True, only_det=True)
config.flag = 'disco'
config_global = ConfigGlobal('train', binary=True, only_det=True)

V2XSIM = '../v2x-sim-1.0/test'

###############################
# numerical checking functions
###############################
def check_attacker_match_cost(match_costs: List, n_agent: int)-> List[bool]:
    """
        Check if the attacker's match cost is the largest.
        Input n_agent

        Check result:

            n_agent: 2, is_max: 1000 / 1000 = 100.00%
            n_agent: 3, is_max: 767 / 900 = 85.22%
            n_agent: 4, is_max: 1177 / 1200 = 98.08%
    """
    match_costs = np.array(match_costs)  # (n_agent - 1, n_agent)
    if match_costs.ndim == 3:
        match_costs = match_costs[:, 0, :]  # (n_agent - 1, n_agent - 1)
    max_cost_agent = match_costs.argmax(axis=0)  # (n_agent, )
    return (max_cost_agent == 0).tolist()

eval_worker = EvalWorker([0.5, ], 1, cache=False)

def check_attack_success(bboxes: List, gt: List, n_agent: int, is_max: List)-> List[bool]:
    mAP_d = eval_worker.evaluate([get_pred_box(b) for b in bboxes[0]], gt, n_agent)
    no_com_mAP = [mAP_d[f"agent_{i} mAP@0.5"] for i in range(n_agent)]

    mAP_d = eval_worker.evaluate([get_pred_box(b) for b in bboxes[1]], gt, n_agent)
    att_mAP = [mAP_d[f"agent_{i} mAP@0.5"] for i in range(n_agent)]

    return list(zip(no_com_mAP, att_mAP, is_max))

#################################
# functions for draw figures
#################################
def rescale_bboxes(voxel_size, area_extents, bboxes):
    bboxes = bboxes.reshape(-1, 4, 2)  
    bboxes = bboxes.copy()  # copy
    bboxes[..., 0] = (bboxes[..., 0] - area_extents[0][0]) / voxel_size[0]
    bboxes[..., 1] = (bboxes[..., 1] - area_extents[1][0]) / voxel_size[1]
    return bboxes

def bev_draw(bev, overlap=None):
    img = np.ones(bev.shape[:2] + (3, )) * np.array([10, 10, 10]) / 255
    if overlap is not None:
        img[overlap] = np.array([50, 50, 50]) / 255
    img[bev > 0] = np.array([255, 255, 255]) / 255
    
    return img

def basic_box_draw(ax, bev, bboxes_color: List):
    ax.imshow(bev, cmap='gray')
    for bboxes, color in bboxes_color:
        for bbox in bboxes:
            polygon = patches.Polygon(bbox, fill=False, edgecolor=color, linewidth=3.0)
            ax.add_patch(polygon)

def set_title_and_text(ax, title: str, text: str):
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(text)

def overlap_from_trans(trans: torch.Tensor, size=(256, 256)):
    x_trans = (4*trans[0, 3])/128
    y_trans = -(4*trans[1, 3])/128

    theta_rot = torch.tensor([[trans[0,0], trans[0,1], 0.0], [trans[1,0], trans[1,1], 0.0]]).type(dtype=torch.float)
    theta_rot = torch.unsqueeze(theta_rot, 0)
    grid_rot = F.affine_grid(theta_rot, size=(1, 1) + size)  # for grid sample

    theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float)
    theta_trans = torch.unsqueeze(theta_trans, 0)
    grid_trans = F.affine_grid(theta_trans, size=(1, 1) + size)  # for grid sample

    mask = torch.ones((1, 1) + size)
    warp_mask_rot = F.grid_sample(mask, grid_rot, mode='bilinear')
    warp_mask_trans = F.grid_sample(warp_mask_rot, grid_trans, mode='bilinear')
    warp_mask = torch.squeeze(warp_mask_trans)

    return (warp_mask > 0).numpy()

def visualize(bevs, trans_matrices_list, gt_box, det_boxes, n_agent, mAPs, match_costs, sample_name, prefix='', vis_att=False):
    # TODO: draw confidence score in figure
    voxel_size = config.voxel_size
    area_extents = config.area_extents

    for i in range(n_agent):
        fig = plt.figure(figsize=(7*(n_agent+1), 8))
        axes = fig.subplots(ncols=n_agent+1)
        fig.suptitle(f"{sample_name}")

        set_title_and_text(axes[0], "Ground Truth", "")
        gts = rescale_bboxes(voxel_size, area_extents, gt_box[i]['bboxes'])
        bev_img = bev_draw(bevs[i])
        basic_box_draw(axes[0], bev_img, [(gts, '#58ABFE')])

        if len(det_boxes) > n_agent and vis_att:
            set_title_and_text(axes[1], f"Agent {i} Attacked (all agent com)", f"mAP@0.5 = {mAPs[0][i]*100:.2f}")
            preds = rescale_bboxes(voxel_size, area_extents, det_boxes[-1][i][0]['pred'])
            basic_box_draw(axes[1], bev_img, [(gts, 'r'), (preds, '#00FA65')])
        else:
            set_title_and_text(axes[1], f"Agent {i} Self", f"mAP@0.5 = {mAPs[0][i]*100:.2f}")
            preds = rescale_bboxes(voxel_size, area_extents, det_boxes[0][i][0]['pred'])
            basic_box_draw(axes[1], bev_img, [(gts, 'r'), (preds, '#00FA65')])

        for j in range(1, n_agent):
            set_title_and_text(axes[j+1], f"Agent {i}, {(i+j) % n_agent}", f"mAP@0.5 = {mAPs[j][i]*100:.2f} match_cost = {match_costs[j-1][i]:.2f}")
            preds = rescale_bboxes(voxel_size, area_extents, det_boxes[j][i][0]['pred'])
            overlap = overlap_from_trans(trans_matrices_list[i][0, (i+j) % n_agent])
            bev_img = bev_draw(bevs[i], overlap)
            basic_box_draw(axes[j+1], bev_img, [(gts, 'r'), (preds, '#00FA65')])

        legend_elements = [patches.Patch(edgecolor='#58ABFE', facecolor='w', label="Ground Truth Box"),
                    patches.Patch(edgecolor='r', facecolor='w', label="Ground Truth Box"),
                    patches.Patch(edgecolor='#00FA65', facecolor='w', label="Predicted Box"),
                    patches.Patch(facecolor='#565656', edgecolor='#0A0A0A', label="Overlapped Region"),]        
        
        fig.legend(handles=legend_elements, loc='lower right', shadow=True, fontsize='x-large') 

        fig.savefig(f"{os.path.join(prefix, sample_name)}_{i}.png")
        plt.close(fig)
        plt.close(fig)

def simple_visualize(bevs, gt_box, det_boxes, n_agent, sample_name, prefix=''):
    voxel_size = config.voxel_size
    area_extents = config.area_extents

    fig = plt.figure(figsize=(7*n_agent, 8))
    axes = fig.subplots(ncols=n_agent)

    for i in range(n_agent):
        gts = rescale_bboxes(voxel_size, area_extents, gt_box[i]['bboxes'])
        preds = rescale_bboxes(voxel_size, area_extents, det_boxes[i][0][:, :-1])
        bev_img = bev_draw(bevs[i])
        # [(gts, '#00ff00'), (preds, '#ff0000')]
        basic_box_draw(axes[i], bev_img, [(preds, '#ff0000')])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    fig.savefig(f"{os.path.join(prefix, sample_name)}.png")
    plt.close(fig)
    plt.close(fig)

def simple_visualize_per_agent(bevs, gt_box, det_boxes, n_agent, sample_name):
    voxel_size = config.voxel_size
    area_extents = config.area_extents

    for i in range(n_agent):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        gts = rescale_bboxes(voxel_size, area_extents, gt_box[i]['bboxes'])
        preds = rescale_bboxes(voxel_size, area_extents, det_boxes[i][0][:, :-1])
        bev_img = bev_draw(bevs[i])
        basic_box_draw(ax, bev_img, [(gts, '#00ff00'), (preds, '#ff0000')])
        ax.set_xticks([])
        ax.set_yticks([])

        prefix = os.path.join(args.dir, f"agent_{i}")
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        fig.savefig(os.path.join(prefix, f"{sample_name}.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

#####################
# main functions
#####################
def check_largest_match_cost(args: argparse.Namespace):
    # Load the results
    # result_dir = "./experiments/match_costs_fix/"
    # result_files = glob.glob(os.path.join(result_dir, "*.pkl"))
    result_dir = args.dir
    result_files = glob.glob(os.path.join(result_dir, "match_costs", "*.pkl"))
    result_files.sort()

    if args.tofile:
        fout = open(os.path.join(result_dir, "match_cost.txt"), 'a')
        fout.write("\n")  # TODO add date here
    
    n_agent_dict = {
        2: [],
        3: [],
        4: [],
    }

    flatten_match_costs = np.array([])
    flatten_attack_labels = np.array([])
    # TODO: 
    # 1. 对不同n_agent的结果分别分析
    # 2. 检查attacker的match cost是否是最大的
    # 3. plot match cost的分布
    # 4. 对match cost进行排序，检查差值是否有规律
    for f in tqdm(result_files):
        with open(f, "rb") as f_in:
            result = pickle.load(f_in)
        bboxes = result["bboxes"]
        match_costs = result["match_costs"]
        gt_box = result["gt_box"]

        n_agents = len(bboxes[0])
        
        is_max = check_attacker_match_cost(match_costs, n_agents)
        mAPs = check_attack_success(bboxes, gt_box, n_agents, is_max)
        success = np.array([mAP[0] - mAP[1] > 0.05 for mAP in mAPs])
        n_agent_dict[n_agents].extend(is_max)

        match_costs_np = np.array(match_costs)
        match_costs_np = match_costs_np[:, 0] if match_costs_np.ndim == 3 else match_costs_np

        attacker_labels = np.zeros_like(match_costs_np)
        attacker_labels[0, success] = 1
        flatten_match_costs = np.concatenate([flatten_match_costs, match_costs_np.flatten()])
        flatten_attack_labels = np.concatenate([flatten_attack_labels, attacker_labels.flatten()])

        if args.tofile and n_agents > 2:
            fout.write(f.split('/')[-1].split('.')[0] + '\n')
            for mAP in mAPs:
                fout.write(f"{mAP[0]*100:.2f} {mAP[1]*100:.2f} {mAP[2]}\n")

    for k in [2, 3, 4]:
        total = len(n_agent_dict[k])
        is_max = sum(n_agent_dict[k])
        print(f"n_agent: {k}, is_max: {is_max} / {total} = {is_max / total * 100:.2f}%")
        if args.tofile:
            fout.write(f"n_agent: {k}, is_max: {is_max} / {total} = {is_max / total * 100:.2f}%\n")
    
    # ROC
    fpr, tpr, thresholds = metrics.roc_curve(flatten_attack_labels, flatten_match_costs)
    roc_auc = metrics.auc(fpr, tpr)
    print(f"AUC: {roc_auc}")
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    #                                 estimator_name='example estimator')
    # display.plot()
    # plt.savefig(os.path.join(result_dir, "roc.png"))

    # Histogram
    flatten_attack_labels = flatten_attack_labels.astype(np.bool8)
    bins = np.histogram(flatten_match_costs, bins=100)[1]
    plt.cla()
    plt.hist(flatten_match_costs[~flatten_attack_labels], bins=bins, color='b', stacked=True, label="normal")
    plt.hist(flatten_match_costs[flatten_attack_labels], bins=bins, color='r', stacked=True, label='attacked')
    plt.legend()
    plt.savefig(os.path.join(result_dir, "hist.png"))

    if args.tofile:
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
        display.plot()
        plt.savefig(os.path.join(result_dir, "roc.png"))
        fout.close()

def draw(args: argparse.Namespace = None):
    if args:
        # result_dir = os.path.join(args.dir, 'match_costs')
        result_dir = args.dir
    else:
        result_dir = "./experiments/match_costs_fix/"
    valset = valset = V2XSIMDataset(dataset_roots=[f'{V2XSIM}/agent{i}' for i in range(5)], config=config, config_global=config_global, split='val', val=True)
    range_4 = list(range(200, 300)) + list(range(600, 700)) + list(range(800, 900))
    valset = torch.utils.data.Subset(valset, range_4)
    loader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    for cnt, sample in tqdm(enumerate(loader), total=len(loader)):
        padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, gt_max_iou, filenames, \
        target_agent_id_list, num_agent_list, trans_matrices_list = zip(*sample)

        padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        sample_name = filenames[0][0][0].split('/')[-2]
        n_agent = num_agent_list[0].item()
        bevs = padded_voxel_points[:n_agent, 0].cpu().numpy().max(axis=-1)

        result_file = os.path.join(result_dir, f"{sample_name}.pkl")
        with open(result_file, 'rb') as f:
            result_dict = pickle.load(f)
        import ipdb;ipdb.set_trace()
        gt_box = result_dict["gt_box"]
        bboxes = result_dict["bboxes"]
        match_costs = result_dict["match_costs"]

        match_costs = np.array(match_costs)
        if match_costs.ndim == 3:
            match_costs = match_costs[:, 0, :]
        
        mAPs = []
        for i in range(n_agent):
            rd = eval_worker.evaluate([get_pred_box(b) for b in bboxes[i]], gt_box, n_agent)
            mAPs.append([rd[f"agent_{j} mAP@0.5"] for j in range(n_agent)])
        
        if args is None or args.vis_att:
            prefix = os.path.join(result_dir, f"{n_agent}_agents")
        else:
            # prefix = os.path.join(result_dir, f"{n_agent}_agents")
            prefix = os.path.join(args.dir, f"{n_agent}_agents")
            
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        # prefix = ''
        visualize(bevs, trans_matrices_list, gt_box, bboxes, n_agent, mAPs, match_costs, sample_name, prefix, vis_att=args.vis_att)
        # break

def auc_only(args: argparse.Namespace = None, ax=None, name=None):
    result_dir = args.dir
    result_files = glob.glob(os.path.join(result_dir, "match_costs", "*.pkl"))
    result_files.sort()

    flatten_match_costs = np.array([])
    flatten_attack_labels = np.array([])

    for f in tqdm(result_files):
        with open(f, "rb") as f_in:
            result = pickle.load(f_in)
        bboxes = result["bboxes"]
        match_costs = result["match_costs"]
        gt_box = result["gt_box"]

        n_agents = len(bboxes[0])
        
        is_max = check_attacker_match_cost(match_costs, n_agents)
        mAPs = check_attack_success(bboxes, gt_box, n_agents, is_max)
        success = np.array([mAP[0] - mAP[1] > 0.05 for mAP in mAPs])

        match_costs_np = np.array(match_costs)
        match_costs_np = match_costs_np[:, 0] if match_costs_np.ndim == 3 else match_costs_np

        attacker_labels = np.zeros_like(match_costs_np)
        attacker_labels[0, success] = 1
        flatten_match_costs = np.concatenate([flatten_match_costs, match_costs_np.flatten()])
        flatten_attack_labels = np.concatenate([flatten_attack_labels, attacker_labels.flatten()])

    # ROC
    fpr, tpr, thresholds = metrics.roc_curve(flatten_attack_labels, flatten_match_costs)
    roc_auc = metrics.auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:0.2f})")

    xlabel = "False Positive Rate"
    ylabel = "True Positive Rate"
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="lower right")
    return ax, (fpr, tpr, roc_auc)

def simple_draw(args: argparse.Namespace):
    # result_dir = os.path.join(args.dir, 'box')
    result_dir = args.dir

    valset = valset = V2XSIMDataset(dataset_roots=[f'{V2XSIM}/agent{i}' for i in range(5)], config=config, config_global=config_global, split='val', val=True)
    range_4 = list(range(200, 300)) + list(range(600, 700)) + list(range(800, 900))
    valset = torch.utils.data.Subset(valset, range_4)
    loader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    for cnt, sample in tqdm(enumerate(loader), total=len(loader)):
        padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, gt_max_iou, filenames, \
        target_agent_id_list, num_agent_list, trans_matrices_list = zip(*sample)

        padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        sample_name = filenames[0][0][0].split('/')[-2]
        n_agent = num_agent_list[0].item()
        bevs = padded_voxel_points[:n_agent, 0].cpu().numpy().max(axis=-1)

        result_file = os.path.join(result_dir, f"{sample_name}.pkl")
        with open(result_file, 'rb') as f:
            result_dict = pickle.load(f)
        pred_bbox = result_dict['pred_bbox']
        gt_bbox = result_dict['gt_bbox']
        # import ipdb;ipdb.set_trace()
        prefix = os.path.join(args.dir, f"{n_agent}_agents")
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        # import ipdb;ipdb.set_trace()
        # simple_visualize(bevs, gt_bbox, pred_bbox, n_agent, sample_name, prefix)
        simple_visualize_per_agent(bevs, gt_bbox, pred_bbox, n_agent, sample_name)
        # import ipdb;ipdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="result save directory")
    parser.add_argument("--tofile", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis_att", action="store_true")

    args = parser.parse_args()
    return args 

# python result_analysis.py experiments/visualize/single_agent_attack/ --vis_att

# python result_analysis.py experiments/visualize/no_attack_single/ --vis

# python result_analysis.py experiments/visualize/ego_single_real/ --vis

# python result_analysis.py experiments/visualize/multi_test_v3_N01_E2e-01_S10/ --vis

# python result_analysis.py experiments/visualize/N01_E1e-01_S10_vis --vis
# python result_analysis.py experiments/visualize/N01_E5e-02_S10_vis --vis

# python result_analysis.py experiments/visualize/oracle_N01_E2e-01_S10 --vis

if __name__ == "__main__":
    # draw()
    # exit()
    args = parse_args()
    simple_draw(args)
    # exit()
    # check_largest_match_cost(args)
    # if args.vis:
    #     draw(args)
    exit()


    ############
    # for auc plot
    ############
    args = parse_args()
    # auc_only(args)
    ax = None
    reuslt_list = []
    for d in ["experiments/pgd_alpha_1e-1_step_10_eps_5e-1/",
            "experiments/pgd_alpha_1e-2_step_10_eps_5e-2/",
            "experiments/pgd_alpha_2e-2_step_10_eps_1e-1/",]:
        args.dir = d
        ax, r = auc_only(args, ax, name=f"eps = {d.split('/')[-2].split('_')[-1]}")
        reuslt_list.append(r)
    plt.title("ROC")
    plt.savefig("auc.png")