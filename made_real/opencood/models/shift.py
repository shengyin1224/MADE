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
from opencood.utils.box_utils import corner_to_center_torch
from torchattacks.attack import Attack
import matplotlib.pyplot as plt
import math

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
import matplotlib.pyplot as plt
from opencood.utils.box_to_feature import boxes_to_feature_grid
import os 
import sys 
import ipdb
import cv2 as cv
import time

np.random.seed(2018)

class Shift(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        bbox_num (int): number of bounding box which will be shifted
        shift_direction (str): "up","down","left","right","random" (Default: "random")
        shift_length (int): how much grids to shift
        padding_type (str): how to fill the origin grid, "zero","around" (Default: "around")
        n_att (int): number of attackers

    """

    def __init__(self, model, fuse_model, record_len, t_matrix, pairwise_t_matrix, bbox_num, shift_length, n_att = 1, shift_direction = "random", padding_type = "around", attack_target = 'pred', save_attack = True, att_layer = [0,1,2]):
        super().__init__("Shift", model)
        """
            需要支持的功能：
            1. attacker个数 -- to do
            2. targeted 攻击？ -- to do
            3. colla_attack -- to do 
            4. 攻击gt box还是pred box
        """
        self.bbox_num = bbox_num
        self.shift_length = shift_length
        self.shift_direction = shift_direction
        self.padding_type = padding_type
        self.n_att = n_att
        self.fuse_model = fuse_model
        self.record_len = record_len
        self.t_matrix = t_matrix
        self.pairwise_t_matrix = pairwise_t_matrix
        self.att_layer = att_layer

    def get_shift_direction_and_grid(self, attacked_grid_list, shift_dir_of_box, bbox_num=None):
        """
        输入每个被攻击的box的坐标,返回每个box的移动方向和(x,y)方便索引
        attacked_grid_list: [[[x],[y]],[[x],[y]],...]
        """
        box_point_num = []
        box_directions = []
        box_grid_x = []
        box_grid_y = []
        direction_list = ["up","down","left","right"]

        if bbox_num != None:
            tmp_bbox_num  = bbox_num
        else:
            tmp_bbox_num  = self.bbox_num

        # print(len(shift_dir_of_box))
        for i in range(tmp_bbox_num):
            tmp_k = len(attacked_grid_list[i][0])
            if self.shift_direction in direction_list:
                box_direction = [direction_list.index(self.shift_direction) for j in range(tmp_k)]
            else:
                # import ipdb; ipdb.set_trace()
                tmp_dir = shift_dir_of_box[i]
                box_direction = [tmp_dir for j in range(tmp_k)]
            box_directions.extend(box_direction)
            box_grid_x.extend(attacked_grid_list[i][0])
            box_grid_y.extend(attacked_grid_list[i][1])
            if i == 0:
                box_point_num.append(tmp_k)
            else:
                box_point_num.append(tmp_k + box_point_num[i-1])
        return box_grid_x, box_grid_y, box_directions, box_point_num
    
    def get_erase_point_num_and_grid(self, erase_grid_list, erase_index):
        """
        输入每个被攻击的box的坐标,返回每个box的移动方向和(x,y)方便索引
        attacked_grid_list: [[[x],[y]],[[x],[y]],...]
        """
        box_grid_x = []
        box_grid_y = []
        box_point_num = []

        for i in range(len(erase_index)):
            tmp_k = len(erase_grid_list[i][0])
            box_grid_x.extend(erase_grid_list[i][0])
            box_grid_y.extend(erase_grid_list[i][1])
            if i == 0:
                box_point_num.append(tmp_k)
            else:
                box_point_num.append(tmp_k + box_point_num[i-1])
        return box_grid_x, box_grid_y, box_point_num

    def get_erase_grid(self, erase_grid_list, erase_index):

        box_grid_x = []
        box_grid_y = []
        for i in range(len(erase_index)):
            tmp_k = len(erase_grid_list[i][0])
            box_grid_x.extend(erase_grid_list[i][0])
            box_grid_y.extend(erase_grid_list[i][1])
        return box_grid_x, box_grid_y

    def where_line(self,A,B,M):
        """
        A,B 表示有向线段的两端
        M 表示某个点
        最终返回 1表示M点在线段的左侧, 0表示在线段上, -1表示在右侧
        """
        Ax, Ay = A[0], A[1]
        Bx, By = B[0], B[1]
        X, Y = M[0], M[1]
        position = torch.sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
        return position

    def model_run(self, data_dict, num = 0, attack_target = 'pred', 
                  shift_feature = False, rotate_feature = False, attack_conf = None, 
                  real_data_dict = None, attack_srcs = [], if_erase = False, erase_index = [], 
                  dataset = None, pred_gt_box_tensor = None, shift_dir_of_box = [], attack = None, if_fuse = True, if_inference = False, attacked_feature = False, gt_box_tensor=None,
                  cls = None):

        """
        由于where2comm下的结构比较复杂,所以这里单独构造一个函数跑函数结果
        """

        feature_list = data_dict['feature_list']

        # if self.multi_scale:
        if attacked_feature:
            fused_feature, residual_vector = self.fuse_model(feature_list,
                                        self.record_len,
                                        self.t_matrix, 
                                        data_dict = data_dict, attack = attack, 
                                        attack_src = attack_srcs, num = num, 
                                        shift_feature = shift_feature, rotate_feature = rotate_feature,
                                        attack_conf = attack_conf, real_data_dict = real_data_dict, if_erase = if_erase, 
                                        erase_index = erase_index, attack_target = attack_target, 
                                        pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, 
                                        shift_dir_of_box = shift_dir_of_box, if_fuse = if_fuse, if_inference = if_inference, attacked_feature = attacked_feature, gt_box_tensor = gt_box_tensor)
        else:
            fused_feature, residual_vector, _ = self.fuse_model(feature_list,
                                        self.record_len,
                                        self.t_matrix, 
                                        data_dict = data_dict, attack = attack, 
                                        attack_src = attack_srcs, num = num, 
                                        shift_feature = shift_feature, rotate_feature = rotate_feature,
                                        attack_conf = attack_conf, real_data_dict = real_data_dict, if_erase = if_erase, 
                                        erase_index = erase_index, attack_target = attack_target, 
                                        pred_gt_box_tensor = pred_gt_box_tensor, dataset = dataset, 
                                        shift_dir_of_box = shift_dir_of_box, if_fuse = if_fuse, if_inference = if_inference, attacked_feature = attacked_feature, gt_box_tensor = gt_box_tensor, if_att_score
                                        = True, cls = cls)
        
        return fused_feature, residual_vector

    def if_in_box(self, corner_list, point_grid):
        """
        2 -------- 1
        |          |
        |          |
        |          |
        3 -------- 0

        inputs:
            corner_list: (4, 2) 表示box四个点的坐标,顺序是0,1,2,3
            point_grid: (2, ) 表示某个点的坐标
        outputs:
            flag: True/False 表示是否在box中 
        """
        flag = True
        tmp_1, tmp_2 = [0,1,2,3], [1,2,3,0]
        point_grid = torch.tensor(point_grid)
        for i in range(4):
            tmp = self.where_line(corner_list[tmp_1[i]],corner_list[tmp_2[i]],point_grid)
            # tmp >= 0都是可以的
            if tmp < 0:
                flag = False
                break
        
        return flag

    def change_view_bbox(self, bbox_tensor):

        N = bbox_tensor.shape[0]
        add_part = torch.ones(size=(N, 8, 1)).cuda()
        padding_tensor = torch.cat((bbox_tensor, add_part), dim=2)
        
        new_tensor = torch.matmul(padding_tensor, self.pairwise_t_matrix[0][0,1].float().T)

        return new_tensor[:,:,[0,1,2]] 

    def draw_warp_feature_map(self, pred_gt_box_tensor, num, shift_dir_of_box):

        tensor = torch.zeros(size=(100,252))
        attacked_bbox_center = corner_to_center_torch(corner3d=pred_gt_box_tensor, order='hwl')
        attacked_bbox_grid, other_information_list, corner_list  = boxes_to_feature_grid(attacked_bbox_center)

        tensor1 = torch.zeros(size=(100,252))
        warp_box_tensor = self.change_view_bbox(pred_gt_box_tensor)
        attacked_bbox_center1 = corner_to_center_torch(corner3d=warp_box_tensor, order='hwl')
        attacked_bbox_grid1, other_information_list1, corner_list1  = boxes_to_feature_grid(attacked_bbox_center1)

        bbox_num = attacked_bbox_center.shape[0]
        grid_x, grid_y = self.get_grids_in_box(bbox_num, corner_list, shift_dir_of_box)
        grid_x1, grid_y1 = self.get_grids_in_box(bbox_num, corner_list1, shift_dir_of_box)

        tensor[grid_y,grid_x] = 1
        tensor1[grid_y1,grid_x1] = 1

        # 创建图形和子图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(tensor)
        axes[0].set_title('ego agent view')

        axes[1].imshow(tensor1)
        axes[1].set_title('ISU view')

        # 调整子图之间的间距
        plt.subplots_adjust(wspace=0.3)

        save_path = '/GPFS/data/shengyin/OpencoodV2-Main/OpenCOODv2/outcome/feature_save/shift_view'
        plt.savefig(save_path + f'/sample_{num}.png')
        plt.close()


    def get_grids_in_box(self, bbox_num, corner_list, shift_dir_of_box):

        attacked_grid_list = []
        for k in range(bbox_num):
            tmp_grid_list = [[],[]]
            tmp_corner_list = corner_list[k] # (4, 2)
            max_x, min_x, max_y, min_y = max(tmp_corner_list[:,0]), min(tmp_corner_list[:,0]), max(tmp_corner_list[:,1]), min(tmp_corner_list[:,1])
            max_x, min_x, max_y, min_y = max_x.int(), min_x.int(), max_y.int(), min_y.int()
            for x_i in range(min_x, max_x + 1):
                for y_i in range(min_y, max_y + 1):
                    flag = self.if_in_box(tmp_corner_list, [x_i, y_i])
                    if flag:
                        tmp_grid_list[0].append(x_i)
                        tmp_grid_list[1].append(y_i)

            attacked_grid_list.append(tmp_grid_list)
        
        grid_x, grid_y, shift_dir, point_num = self.get_shift_direction_and_grid(attacked_grid_list, shift_dir_of_box = shift_dir_of_box, bbox_num=bbox_num)
        
        return grid_x, grid_y

         

    def forward(self, data_dict, batch_dict, if_attack_feat = False, attack_feat = None, attack_srcs = [], if_erase = False, erase_index = [], num = 0, attack_conf = None, attack_target = 'pred', pred_gt_box_tensor = None, dataset = None, shift_dir_of_box = [], erase_value = None, gt_box_tensor = None, att_layer = [0, 1, 2]):
        r"""
        Shift some bboxes to attack.
        """


        # only one agent
        if batch_dict['spatial_features'].shape[0] - 1 < self.n_att:
            return [torch.zeros(64, 100, 252).cuda(), torch.zeros(128, 50, 126).cuda(), torch.zeros(256, 25, 63).cuda()], attack_srcs

        if if_erase:
            erase_padding_type = attack_conf.attack.erase.padding_type
            do_erase = attack_conf.attack.erase.do_erase

        if attack_target == 'gt':
            # select attacked bboxes and compute corresponding grids
            box_tensor = self.change_view_bbox(gt_box_tensor)
            object_bbx_center = corner_to_center_torch(corner3d=box_tensor, order='hwl')

            if self.bbox_num == 'max':
                self.bbox_num = object_bbx_center.shape[0]
            attacked_bbox_center = object_bbx_center[:self.bbox_num]

            # shape: (20, 2) -- x, y and x > y ; (20, 2) -- l, w and l > w ; (20, 4, 2)  
            attacked_bbox_grid, other_information_list, corner_list  = boxes_to_feature_grid(attacked_bbox_center) 
        else:
            # change the view of grids
            box_tensor = self.change_view_bbox(pred_gt_box_tensor)

            object_bbx_center = corner_to_center_torch(corner3d=box_tensor, order='hwl')
            
            if self.bbox_num == 'max':
                self.bbox_num = object_bbx_center.shape[0]
            attacked_bbox_center = object_bbx_center[:self.bbox_num]
            # shape: (20, 2) -- x, y and x > y ; (20, 2) -- l, w and l > w ; (20, 4, 2)  
            attacked_bbox_grid, other_information_list, corner_list  = boxes_to_feature_grid(attacked_bbox_center)

        # 对三层分别求corner_list
        all_corner_list = []
        all_corner_list.append(corner_list)
        for layer in range(2):
            tmp_list = []
            for box in corner_list:
                tmp_list.append(torch.floor(box / math.pow(2, layer+1)))
            all_corner_list.append(tmp_list)

        if if_erase:
            box_index = []
            for j in range(object_bbx_center.shape[0]):
                if j not in erase_index:
                    box_index.append(j)

        # get all points in the box (or bigger area)
        all_attacked_grid_list = []
        all_erase_grid_list = []
        for layer in range(3):

            corner_list = all_corner_list[layer]

            attacked_grid_list = []
            erase_grid_list = []
            for k in range(self.bbox_num):
                tmp_grid_list = [[],[]]
                tmp_corner_list = corner_list[k] # (4, 2)
                max_x, min_x, max_y, min_y = max(tmp_corner_list[:,0]), min(tmp_corner_list[:,0]), max(tmp_corner_list[:,1]), min(tmp_corner_list[:,1])
                max_x, min_x, max_y, min_y = max_x.int(), min_x.int(), max_y.int(), min_y.int()
                for x_i in range(min_x, max_x + 1):
                    for y_i in range(min_y, max_y + 1):
                        flag = self.if_in_box(tmp_corner_list, [x_i, y_i])
                        if flag:
                            tmp_grid_list[0].append(x_i)
                            tmp_grid_list[1].append(y_i)
                
                if not if_erase or k in box_index:
                    attacked_grid_list.append(tmp_grid_list)
                else:
                    erase_grid_list.append(tmp_grid_list)
            
            all_attacked_grid_list.append(attacked_grid_list)
            all_erase_grid_list.append(erase_grid_list)

            

        if if_erase:
            self.bbox_num -= len(erase_index)
            all_erase_corner_list = []
            all_shift_corner_list = []
            for layer in range(3):
                corner_list = all_corner_list[layer]
                if len(corner_list) == 0:
                    tmp_corner_list = torch.tensor([]).cuda()
                    all_shift_corner_list.append(tmp_corner_list[box_index])
                    all_erase_corner_list.append(tmp_corner_list[erase_index])
                    continue
                tmp_corner_list = torch.stack(corner_list)
                all_shift_corner_list.append(tmp_corner_list[box_index])
                all_erase_corner_list.append(tmp_corner_list[erase_index])
            
        # get attacked grid and processed grid
        all_point_num, all_erase_point_num = [], []
        all_grid_x, all_grid_y, all_processed_grid_x, all_processed_grid_y = [], [], [], []
        all_erase_x, all_erase_y = [], []
        H_list = [100, 50, 25]
        W_list = [252, 126, 63]
        for layer in range(3):
            H, W = H_list[layer], W_list[layer]
            
            attacked_grid_list = all_attacked_grid_list[layer]
            erase_grid_list = all_erase_grid_list[layer]
            grid_x, grid_y, shift_dir, point_num = self.get_shift_direction_and_grid(attacked_grid_list, shift_dir_of_box = shift_dir_of_box)
            erase_x, erase_y, erase_point_num = self.get_erase_point_num_and_grid(erase_grid_list, erase_index)
            assert len(grid_x) == len(shift_dir)
            processed_grid_x, processed_grid_y = self.get_processed_grid(grid_x, grid_y, shift_dir, H, W)

            all_point_num.append(point_num)
            all_erase_point_num.append(erase_point_num)
            all_grid_x.append(grid_x)
            all_grid_y.append(grid_y)
            all_processed_grid_x.append(processed_grid_x)
            all_processed_grid_y.append(processed_grid_y)
            all_erase_x.append(erase_x)
            all_erase_y.append(erase_y)

        # 进行膨胀
        # if if_erase:
        #     painter = np.zeros(shape=(100,252))
        #     painter[erase_y, erase_x] = 1
        #     painter = painter.reshape(100,252,1)
        #     kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
        #     painter = cv.dilate(painter, kernel=kernel)
        #     erase_y, erase_x = (np.where(painter > 0)[0]).tolist(), (np.where(painter > 0)[1]).tolist()

        # get attacked feature
        attack_list = []
        attacks = []

        if if_attack_feat:
            total_attacked_feature_list = attack_feat
        else:
            _, total_attacked_feature_list = self.model_run(batch_dict, attack_srcs = attack_srcs , attacked_feature = True)  # ((64,100,252),(128,50,126),(256,25,63))
        
        for layer in range(3):

            C, H, W = total_attacked_feature_list[layer][0].shape
            point_num = all_point_num[layer]
            erase_point_num = all_erase_point_num[layer]
            grid_x = all_grid_x[layer]
            grid_y = all_grid_y[layer]
            processed_grid_x = all_processed_grid_x[layer]
            processed_grid_y = all_processed_grid_y[layer]
            erase_x = all_erase_x[layer]
            erase_y = all_erase_y[layer]
            corner_list = all_shift_corner_list[layer]
            erase_corner_list = all_erase_corner_list[layer]
            attack_list = []
            if layer not in att_layer and not if_erase:
                attacks.append(torch.zeros(self.n_att, C, H, W).cuda())
            else:
                attacked_feature_list = total_attacked_feature_list[layer]
                for i in range(self.n_att):
                    attacked_feature = attacked_feature_list[i]  # (C,H,W)
                    new_attacked_feature = attacked_feature.clone()
                    origin_value_array = attacked_feature[:,grid_y,grid_x] #(64, 20) 
                    
                    # 先抹去范围外的bbox
                    if if_erase and do_erase and layer == 0:
                        if erase_padding_type == "zero":
                            new_attacked_feature[:,erase_y,erase_x] = 0
                        elif erase_padding_type == "around":
                            
                            # 这里输入和右下角的点相对的点（必定在box外）
                            tmp_grid_x = [erase_corner_list[j][0][0].int() + 1 for j in range(len(erase_index))]
                            tmp_grid_y = [erase_corner_list[j][0][1].int() - 1 for j in range(len(erase_index))]
                            around_value = self.get_around_value(attacked_feature, tmp_grid_x, tmp_grid_y, erase_point_num)
                            new_attacked_feature[:,erase_y,erase_x] = around_value
                        elif erase_padding_type == "mode":
                            if erase_value == None:
                                tmp_x_att = attacked_feature.clone().view(C, -1)
                                value, count = torch.unique(tmp_x_att, dim=1, return_counts = True)
                                max_index = torch.argmax(count)
                                erase_value = value[:,max_index]
                            # np.save(f'outcome/erase_value/sample_{num}.npy', erase_value.detach().cpu().numpy())
                            new_attacked_feature[:,erase_y,erase_x] = erase_value.reshape(64,-1)
                            erase_value = None
                    
                    if layer in att_layer:
                        # shift 
                        if processed_grid_x.shape[0] != 0:
                            new_attacked_feature[:,processed_grid_y,processed_grid_x] = origin_value_array
            
                attack_list.append((new_attacked_feature - attacked_feature).cpu().detach().numpy())
                attacks.append(attack_list)
                
        
        return attacks, attack_srcs

    # new grid after shifting 
    def get_processed_grid(self, grid_x, grid_y, shift_dir, H, W, layer=0):
        """
        生成移动后的x,y坐标
        """

        shift_length = self.shift_length / (math.pow(2, layer))

        shift_dir_x = torch.tensor(shift_dir)
        shift_dir_y = torch.clone(shift_dir_x)

        shift_dir_x[shift_dir_x == 0], shift_dir_x[shift_dir_x == 1] = 0, 0
        shift_dir_x[shift_dir_x == 2] = -shift_length
        shift_dir_x[shift_dir_x == 3] = shift_length

        for i in range(shift_dir_y.shape[0]):
            if shift_dir_y[i] == 0:
                shift_dir_y[i] = shift_length
            elif shift_dir_y[i] == 1:
                shift_dir_y[i] = -shift_length
            else:
                shift_dir_y[i] = 0
        
        grid_x = torch.tensor(grid_x) + shift_dir_x
        grid_y = torch.tensor(grid_y) + shift_dir_y

        # 其中 0 =< x < 252, 0 <= y < 100
        grid_x[grid_x < 0], grid_x[grid_x >= W] = 0, W-1  
        grid_y[grid_y < 0], grid_y[grid_y >= H] = 0, H-1
          
        return grid_x, grid_y

    # TODO: 精细化？如何正确选择到框外面的点？
    def get_around_value(self, attacked_feature, grid_x, grid_y, point_num_list, H, W):

        grid_x_0 = [grid_x[i] - 1 for i in range(self.bbox_num)]
        grid_x_1 = [grid_x[i] + 1 for i in range(self.bbox_num)]
        grid_y_0 = [grid_y[i] - 1 for i in range(self.bbox_num)]
        grid_y_1 = [grid_y[i] + 1 for i in range(self.bbox_num)]

        around_value = torch.zeros(size=(attacked_feature.shape[0],point_num_list[self.bbox_num-1])).cuda() # (64, total_point_num)
        for i in range(self.bbox_num):
            if i == 0:
                start_i = 0
            else:
                start_i = point_num_list[i-1]
            end_i = point_num_list[i]
            
            tmp_length = end_i - start_i
            if grid_x_0[i] >= 0 and grid_x_0[i] < W:
                if grid_y_0[i] >= 0 and grid_y_0[i] < H:
                    around_value[:,start_i:end_i] = attacked_feature[:,grid_y_0[i],grid_x_0[i]].repeat(tmp_length,1).T
                    continue
                elif grid_y_1[i] >= 0 and grid_y_1[i] < H:
                    around_value[:,start_i:end_i] = attacked_feature[:,grid_y_1[i],grid_x_0[i]].repeat(tmp_length,1).T
                    continue
            elif grid_x_1[i] >= 0 and grid_x_1[i] < W:
                if grid_y_0[i] >= 0 and grid_y_0[i] < H:
                    around_value[:,start_i:end_i] = attacked_feature[:,grid_y_0[i],grid_x_1[i]].repeat(tmp_length,1).T
                    continue
                elif grid_y_1[i] >= 0 and grid_y_1[i] < H:
                    around_value[:,start_i:end_i] = attacked_feature[:,grid_y_1[i],grid_x_1[i]].repeat(tmp_length,1).T
                    continue
                    
        return around_value

    def inference(self, data_dict, attack, attack_src, delete_list = [], num = 0, if_inference = True, cls = None):

        if delete_list != []:
            if_fuse = False
        else:
            if_fuse = True

        outputs, residual_vector = self.model_run(data_dict, attack = attack, attack_srcs = attack_src, num = num, if_inference = if_inference, if_fuse = if_fuse, cls = cls)

        return outputs, residual_vector

if __name__ == "__main__":
    print("This is a new attack model!")
    