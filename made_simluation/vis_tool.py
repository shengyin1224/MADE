"""
    遍历v2xsim数据集：
        读取其他保存的meta data进行可视化
"""
import os
import glob
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils import data

from data.config import Config, ConfigGlobal
from data.Dataset import V2XSIMDataset

###################
# global variables
###################
config = Config('test', binary=True, only_det=True)
config.flag = 'disco'
config_global = ConfigGlobal('train', binary=True, only_det=True)

V2XSIM = '../v2x-sim-1.0/test'

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("dir", type=str, help="result save directory")
    # parser.add_argument("--tofile", action="store_true")
    # parser.add_argument("--vis", action="store_true")
    # parser.add_argument("--vis_att", action="store_true")

    args = parser.parse_args()
    return args 

def main(args: argparse.Namespace):
    valset = V2XSIMDataset(dataset_roots=[f'{V2XSIM}/agent{i}' for i in range(5)], config=config, config_global=config_global, split='val', val=True)
    valset = torch.utils.data.Subset(valset, range(200, 202))
    loader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    for cnt, sample in tqdm(enumerate(loader), total=len(loader)):
        padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, gt_max_iou, filenames, \
        target_agent_id_list, num_agent_list, trans_matrices_list = zip(*sample)

        padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        sample_name = filenames[0][0][0].split('/')[-2]