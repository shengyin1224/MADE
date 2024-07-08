import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchattacks.attack import Attack

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
import matplotlib.pyplot as plt
from opencood.utils.box_to_feature import boxes_to_feature_grid

class Shift_and_Rotate(Attack):
    r"""
    Arguments:
        model (nn.Module): model to attack.
        bbox_num (int): number of bounding box which will be shifted
        n_att (int): number of attackers

        shift_direction (str): "up","down","left","right","random" (Default: "random")
        shift_length (int): how much grids to shift
        shift_padding_type (str): how to fill the origin grid, "zero","around" (Default: "around")

        shift_angle (int): how much angle to rotate
        shift_type (str): to make sure the exact value of shift angle "exact", "random" (Default: "random")
        shift_padding_type (str): how to fill the origin grid, "zero","around" (Default: "around")

    """

    def __init__(self, model, bbox_num, shift_length, shift_angle, n_att = 1,  shift_direction = "random", shift_padding_type = "around", shift_type = "random", rotate_padding_type = "around"):
        super().__init__("Rotate", model)
        """
            需要支持的功能：
            1. attacker个数 -- to do
            2. targeted 攻击？ -- to do
            3. colla_attack -- to do 
            4. 攻击gt box还是pred box
        """
        self.bbox_num = bbox_num
        self.n_att = n_att

        self.rotate_angle = shift_angle
        self.rotate_padding_type = rotate_padding_type
        self.rotate_type = shift_type

        self.shift_length = shift_length
        self.shift_direction = shift_direction
        self.shift_padding_type = shift_padding_type



    def rotate_get_shift_direction_and_grid(self, attacked_grid_list):
        """
        输入每个被攻击的box的坐标,返回每个box的移动方向和(x,y)方便索引
        attacked_grid_list: [[[x],[y]],[[x],[y]],...]
        """
        box_point_num = []
        box_grid_x = []
        box_grid_y = []
        for i in range(self.bbox_num):
            tmp_k = len(attacked_grid_list[i][0])
            box_grid_x.extend(attacked_grid_list[i][0])
            box_grid_y.extend(attacked_grid_list[i][1])
            if i == 0:
                box_point_num.append(tmp_k)
            else:
                box_point_num.append(tmp_k + box_point_num[i-1])
        return box_grid_x, box_grid_y, box_point_num

    def get_shift_direction_and_grid(self, attacked_grid_list):
        """
        输入每个被攻击的box的坐标,返回每个box的移动方向和(x,y)方便索引
        attacked_grid_list: [[[x],[y]],[[x],[y]],...]
        """
        box_point_num = []
        box_directions = []
        box_grid_x = []
        box_grid_y = []
        direction_list = ["up","down","left","right"]
        for i in range(self.bbox_num):
            tmp_k = len(attacked_grid_list[i][0])
            if self.shift_direction in direction_list:
                box_direction = [direction_list.index(self.shift_direction) for j in range(tmp_k)]
            else:
                tmp_dir = np.random.randint(low=0,high=4)
                box_direction = [tmp_dir for j in range(tmp_k)]
            box_directions.extend(box_direction)
            box_grid_x.extend(attacked_grid_list[i][0])
            box_grid_y.extend(attacked_grid_list[i][1])
            if i == 0:
                box_point_num.append(tmp_k)
            else:
                box_point_num.append(tmp_k + box_point_num[i-1])
        return box_grid_x, box_grid_y, box_directions, box_point_num

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

    def rotate_feature_map(self,rot_angle, input):
        """
        作用是旋转固定的feature map
        """
        B = 1
        input = input.unsqueeze(0)
        angle = rot_angle/180*math.pi
        # 创建一个坐标变换矩阵
        transform_matrix = torch.tensor([
                [math.cos(angle),math.sin(-angle),0],
                [math.sin(angle),math.cos(angle),0]])
        # 将坐标变换矩阵的shape从[2,3]转化为[1,2,3]，并重复在第0维B次，最终shape为[B,2,3]
        transform_matrix = transform_matrix.unsqueeze(0).repeat(B,1,1)

        grid = F.affine_grid(transform_matrix, input.shape)	# 旋转变换矩阵# 变换后的tensor的shape(与输入tensor相同)

        output = F.grid_sample(input, # 输入tensor，shape为[B,C,W,H]
                            grid.cuda(), # 上一步输出的gird,shape为[B,C,W,H]
                            mode='bilinear',padding_mode='reflection') # 一些图像填充方法，这里我用的是最近邻
        return output

    def rotate_bbox(self, tmp_grid_x, tmp_grid_y, attacked_feature, tmp_angle, center_x, center_y):
        """
        tmp_grid_x, tmp_grid_y: 每个bbox在原始feature map上的坐标
        attacked_feature: 原始feature map
        tmp_angle: 旋转的角度
        center_x, center_y: 每个bbox的center坐标
        """
        # 构建白板feature map (64,20,20)，把bbox复制上去
        blank_space = torch.zeros(size=(64,20,20)).cuda()
        index_space = torch.zeros(size=(64,20,20)).cuda()
        grid_x = [tmp_grid_x[i] - center_x + 10 for i in range(len(tmp_grid_x))]
        grid_y = [tmp_grid_y[i] - center_y + 10 for i in range(len(tmp_grid_y))]
        blank_space[:,grid_y,grid_x] = attacked_feature[:,tmp_grid_y,tmp_grid_x]
        index_space[:,grid_y,grid_x] = 1
        # tmp_angle = 90

        # 旋转白板feature map
        new_space = self.rotate_feature_map(rot_angle=tmp_angle, input=blank_space).squeeze(0)
        index_space = self.rotate_feature_map(rot_angle=tmp_angle, input=index_space).squeeze(0)
        index = index_space.nonzero()
        real_index_y = [index[i][1] for i in range(int(len(index) / 64))]
        real_index_x = [index[i][2] for i in range(int(len(index) / 64))]
        index_x, index_y = [], []
        final_index_x, final_index_y = [], []
        for i in range(int(len(index) / 64)):
            tmp_x = index[i][2] - 10 + center_x
            tmp_y = index[i][1] - 10 + center_y
            # 必须在范围内，在范围外的直接排除
            if tmp_x < 352 and tmp_x >= 0 and tmp_y >= 0 and tmp_y < 100:
                index_x.append(tmp_x)
                index_y.append(tmp_y)
                final_index_x.append(index[i][2])
                final_index_y.append(index[i][1])

        return index_x, index_y, new_space[:,final_index_y,final_index_x]

    def update_variable(self, corner_list, attacked_grid_list, attacked_bbox_grid, processed_grid_x, processed_grid_y, point_num, shift_dir):
        """
        corner_list: 之前的corner_list进行平移 (20, 4, 2)
        attacked_grid_list: 直接用processed_x和processed_y和point_num (bbox_num, 2, point_n)
        attacked_bbox_grid: 各个box的center, 直接用之前的corner_list进行平移 (20, 2)
        """

        shift_dir_list = []

        # attacked_grid_list
        new_attacked_grid_list = []
        for i in range(self.bbox_num):

            if i == 0:
                start = 0
            else:
                start = point_num[i-1]
            end = point_num[i]

            shift_dir_list.append(shift_dir[start])

            tmp_x, tmp_y = processed_grid_x[start:end], processed_grid_y[start:end]
            new_attacked_grid_list.append([tmp_x, tmp_y])

        # attacked_bbox_grid
        tmp_x = [attacked_bbox_grid[j][0] for j in range(self.bbox_num)]
        tmp_y = [attacked_bbox_grid[j][1] for j in range(self.bbox_num)]
        new_x, new_y = self.get_processed_grid(tmp_x, tmp_y, shift_dir_list)
        new_attacked_bbox_grid = [[new_x[j],new_y[j]] for j in range(self.bbox_num)]

        # corner_list
        new_corner_list = []
        for i in range(self.bbox_num):
            tmp_x = [corner_list[i][j][0] for j in range(4)]
            tmp_y = [corner_list[i][j][1] for j in range(4)]
            tmp_shift_dir = [shift_dir_list[i] for j in range(4)]
            new_x, new_y = self.get_processed_grid(tmp_x, tmp_y, tmp_shift_dir)
            tmp = [[new_x[j], new_y[j]] for j in range(4)]
            new_corner_list.append(tmp)

        return new_corner_list, new_attacked_grid_list, new_attacked_bbox_grid

    def forward(self, data_dict, batch_dict, if_attack_feat = False, attack_feat = None, attack_srcs = []):
        r"""
        Shift some bboxes to attack.
        """

        # get the attack sources
        attack_srcs = self.get_attack_src(batch_dict['spatial_features'].shape[0])

        # only one agent
        if batch_dict['spatial_features'].shape[0] - 1 < self.n_att:
            return [torch.zeros(64, 100, 352).cuda(), torch.zeros(128, 50, 176).cuda(), torch.zeros(256, 25, 88).cuda()], attack_srcs

        # select attacked bboxes and compute corresponding grids
        object_bbx_center = data_dict['object_bbx_center']
        object_bbx_mask = data_dict['object_bbx_mask']
        object_bbx_center = object_bbx_center[object_bbx_mask == 1] # shape: (20, 7)
        if self.bbox_num == 'max':
            self.bbox_num = object_bbx_center.shape[0]
        attacked_bbox_center = object_bbx_center[:self.bbox_num]

        # shape: (20, 2) -- x, y and x > y ; (20, 2) -- l, w and l > w ; (20, 4, 2)  
        attacked_bbox_grid, other_information_list, corner_list  = boxes_to_feature_grid(attacked_bbox_center) 

        # print(other_information_list)
        # get all points in the box (or bigger area)
        attacked_grid_list = []
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
                        
            attacked_grid_list.append(tmp_grid_list)

        # get attacked grid and processed grid
        # TODO: 这里默认所有attacker攻击的box是相同的
        grid_x, grid_y, shift_dir, point_num = self.get_shift_direction_and_grid(attacked_grid_list)
        assert len(grid_x) == len(shift_dir)
        processed_grid_x, processed_grid_y = self.get_processed_grid(grid_x, grid_y, shift_dir)

        # get attacked feature
        attack_list = []
        
        if if_attack_feat:
            attacked_feature_list = attack_feat
        else:
            _, attacked_feature_list = self.model(batch_dict, attack_src = attack_srcs ,attacked_feature = True) # (n_att,64,100,352)
        
        # Shift
        new_attacked_feature_list = torch.zeros(size=(1,self.n_att,64,100,352)).cuda()
        for i in range(self.n_att):
            attacked_feature = attacked_feature_list[0][i]  # (64,100,352)
            new_attacked_feature = attacked_feature.clone()
            origin_value_array = attacked_feature[:,grid_y,grid_x] #(64, 20)
            
            if self.shift_padding_type == "zero":
                new_attacked_feature[:,grid_y,grid_x] = 0
            elif self.shift_padding_type == "around":
                # 这里每个box选择的点有可能是box内的，这个时候around的点就还是box内的
                # tmp_grid_x = [grid_x[point_num[j]-1] for j in range(len(point_num))]
                # tmp_grid_y = [grid_y[point_num[j]-1] for j in range(len(point_num))]
                
                # 这里输入和右下角的点相对的点（必定在box外）
                tmp_grid_x = [corner_list[j][0][0].int() + 1 for j in range(self.bbox_num)]
                tmp_grid_y = [corner_list[j][0][1].int() - 1 for j in range(self.bbox_num)]
                around_value = self.get_around_value(attacked_feature, tmp_grid_x, tmp_grid_y, point_num)
                new_attacked_feature[:,grid_y,grid_x] = around_value

            # shift
            new_attacked_feature[:,processed_grid_y,processed_grid_x] = origin_value_array
            new_attacked_feature_list[0][i] = new_attacked_feature            

        # TODO: 需要修改 corner_list, attacked_grid_list, attacked_bbox_grid
        # Rotate
        new_corner_list, new_attacked_grid_list, new_attacked_bbox_grid = self.update_variable(corner_list, attacked_grid_list, attacked_bbox_grid, processed_grid_x, processed_grid_y, point_num, shift_dir)

        for i in range(self.n_att):
            tmp_attacked_feature = attacked_feature_list[0][i]  # (64,100,352)
            attacked_feature = new_attacked_feature_list[0][i]  # (64,100,352)
            tmp_new_attacked_feature = attacked_feature.clone()

           # 先填好要被攻击的部分
            if self.rotate_padding_type == "zero":
                tmp_new_attacked_feature[:,processed_grid_y,processed_grid_x] = 0
            elif self.rotate_padding_type == "around":
                # 这里输入和右下角的点相对的点（必定在box外）
                tmp1_grid_x = [new_corner_list[j][0][0].int() + 1 for j in range(self.bbox_num)]
                tmp1_grid_y = [new_corner_list[j][0][1].int() - 1 for j in range(self.bbox_num)]
                around_value = self.get_around_value(attacked_feature, tmp1_grid_x, tmp1_grid_y, point_num)
                tmp_new_attacked_feature[:,processed_grid_y,processed_grid_x] = around_value 
            
            # 再对每一个bbox处理
            for j in range(self.bbox_num):
                tmp_grid_x = new_attacked_grid_list[j][0]
                tmp_grid_y = new_attacked_grid_list[j][1]
                if self.rotate_type == 'exact':
                    tmp_angle = self.rotate_angle
                elif self.rotate_type == 'random':
                    tmp_angle = torch.randint(low=45, high=135, size=(1,))[0]
                center_x, center_y = new_attacked_bbox_grid[j][0], new_attacked_bbox_grid[j][1]

                # 进行旋转，返回旋转后对应部分的grid_x, grid_y和feature
                new_grid_x, new_grid_y, new_bbox = self.rotate_bbox(tmp_grid_x, tmp_grid_y, attacked_feature, tmp_angle, center_x, center_y)
                tmp_new_attacked_feature[:,new_grid_y,new_grid_x] = new_bbox
            
            attack_list.append(tmp_new_attacked_feature - tmp_attacked_feature)


        # get_attack
        attack_list = torch.tensor([item.cpu().detach().numpy() for item in attack_list]).cuda()
        attacks = [attack_list, torch.zeros(self.n_att, 128, 50, 176).cuda(), torch.zeros(self.n_att, 256, 25, 88).cuda()]
        
        return attacks, attack_srcs


        # new grid after shifting 
    def get_processed_grid(self, grid_x, grid_y, shift_dir):
        """
        生成移动后的x,y坐标
        """

        shift_dir_x = torch.tensor(shift_dir)
        shift_dir_y = torch.clone(shift_dir_x)

        shift_dir_x[shift_dir_x == 0], shift_dir_x[shift_dir_x == 1] = 0, 0
        shift_dir_x[shift_dir_x == 2], shift_dir_x[shift_dir_x == 3] = -self.shift_length, self.shift_length

        for i in range(shift_dir_y.shape[0]):
            if shift_dir_y[i] == 0:
                shift_dir_y[i] = self.shift_length
            elif shift_dir_y[i] == 1:
                shift_dir_y[i] = -self.shift_length
            else:
                shift_dir_y[i] = 0
        
        grid_x = torch.tensor(grid_x) + shift_dir_x
        grid_y = torch.tensor(grid_y) + shift_dir_y

        # 其中 0 =< x < 352, 0 <= y < 100
        grid_x[grid_x < 0], grid_x[grid_x >= 352] = 0, 351  
        grid_y[grid_y < 0], grid_y[grid_y >= 100] = 0, 99
          
        return grid_x, grid_y

    # TODO: 精细化？如何正确选择到框外面的点？
    def get_around_value(self, attacked_feature, grid_x, grid_y, point_num_list):

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
            if grid_x_0[i] >= 0 and grid_x_0[i] < 352:
                if grid_y_0[i] >= 0 and grid_y_0[i] < 100:
                    around_value[:,start_i:end_i] = attacked_feature[:,grid_y_0[i],grid_x_0[i]].repeat(tmp_length,1).T
                    continue
                elif grid_y_1[i] >= 0 and grid_y_1[i] < 100:
                    around_value[:,start_i:end_i] = attacked_feature[:,grid_y_1[i],grid_x_0[i]].repeat(tmp_length,1).T
                    continue
            elif grid_x_1[i] >= 0 and grid_x_1[i] < 352:
                if grid_y_0[i] >= 0 and grid_y_0[i] < 100:
                    around_value[:,start_i:end_i] = attacked_feature[:,grid_y_0[i],grid_x_1[i]].repeat(tmp_length,1).T
                    continue
                elif grid_y_1[i] >= 0 and grid_y_1[i] < 100:
                    around_value[:,start_i:end_i] = attacked_feature[:,grid_y_1[i],grid_x_1[i]].repeat(tmp_length,1).T
                    continue
                    
        return around_value

    def inference(self, data_dict, attack, attack_src, delete_list = [], num = 0):        
        outputs, _, _, residual_vector = self.model(data_dict, attack = attack, attack_src = attack_src, num = num, delete_list = delete_list)
        return outputs, residual_vector

if __name__ == "__main__":
    print("This is a new attack model!")
    