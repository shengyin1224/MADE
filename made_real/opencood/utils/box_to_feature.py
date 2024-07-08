import sys
import numpy as np
import torch
import torch.nn.functional as F
from opencood.utils.box_utils import boxes_to_corners2d

# 得到box center的grid坐标
def box_to_feature_grid(attribute):
    """
    attribute: (x,y,z,h,w,l,theta)
    """

    tx, ty = -100.8, -40
    px, py = 0.8, 0.8
    tmp_x, tmp_y = attribute[0], attribute[1]
    x = torch.div((tmp_x - tx), px, rounding_mode='floor')
    y = torch.div((tmp_y - ty), py, rounding_mode='floor')

    return [int(x),int(y)]

# 得到box的长和宽
def box_to_other_information(attribute):
    """
    attribute: (x,y,z,h,w,l,theta)
    """

    tmp_w, tmp_l = attribute[4], attribute[5]
    px, py = 0.8, 0.8

    w = torch.div(tmp_w, py, rounding_mode='floor')
    l = torch.div(tmp_l, px, rounding_mode='floor')
    return [l,w]

# 得到box四个顶点的grid坐标
def get_four_corners(attribte):
    """
    attribute: (x,y,z,h,w,l,theta)
    """
    attribte = torch.as_tensor(attribte).cuda()
    attribte = attribte.unsqueeze(0)  # shape (1,7)
    corners = boxes_to_corners2d(attribte, order= 'hwl')
    corners = corners[:,:,:2].squeeze(0)

    px = 0.8
    tx, ty = -100.8, -40
    corners[:,0] -= tx
    corners[:,1] -= ty
    corners = torch.div(corners, px, rounding_mode='floor')

    for i in range(4):
        corners[i][0] = max(min(251, corners[i][0]), 0)
        corners[i][1] = max(min(99, corners[i][1]), 0)

    return corners

def boxes_to_feature_grid(boxes_center):
    feature_grid_list = []
    corner_list = []
    other_information_list = []
    number = boxes_center.shape[0]
    for i in range(number):
        feature_grid_list.append(box_to_feature_grid(boxes_center[i]))
        other_information_list.append(box_to_other_information(boxes_center[i]))
        corner_list.append(get_four_corners(boxes_center[i]))
    return feature_grid_list, other_information_list, corner_list


# 测试代码
if __name__ == "__main__":
    center = np.load("/GPFS/data/shengyin/OpenCOOD-main/object_bbx_center.npy")
    # center = center[:8]
    print(center)
    print(boxes_to_feature_grid(center))