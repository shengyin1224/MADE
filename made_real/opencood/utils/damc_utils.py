import torch
import numpy as np
from opencood.utils.box_utils import project_box3d
from opencood.utils.common_utils import convert_format, compute_iou

def cal_iou_matrix(cav0_box, cav1_box, transmatrix, ther = 0.3):
        # box0 = cav0_box[:, :4, :2]
        # box1 = cav1_box[:, :4, :2]
        if cav0_box.shape[0] == 0 or cav1_box.shape[0] == 0:
            return [torch.tensor([]), torch.tensor([])]
        cav1_box = project_box3d(cav1_box, transmatrix.float())
        polygons0 = convert_format(cav0_box)
        polygons1 = convert_format(cav1_box)
        iou_matrix = np.zeros((cav0_box.shape[0], cav1_box.shape[0]))
        for i in range(cav0_box.shape[0]):
            iou_matrix[i] = compute_iou(polygons0[i], polygons1[:])
        potential_pair = np.argmax(iou_matrix, 1)
        pair0 = []
        pair1 = []
        # iou = []
        for i in range(len(potential_pair)):
            if iou_matrix[i, potential_pair[i]] > ther:
                pair0.append(i)
                pair1.append(potential_pair[i]) 
                # iou.append(iou_matrix[i, potential_pair[i]])
        return [torch.Tensor(pair0).to(torch.long).cuda(), torch.Tensor(pair1).to(torch.long).cuda()]

def judge_whether(cav0_box, cav1_box, transmatrix):
    return_list = []
    if cav0_box.shape[0] == 0 or cav1_box.shape[0] == 0:
        return [torch.tensor([]), torch.tensor([])]
    cav1_box = project_box3d(cav1_box, transmatrix.float())
    for i in range(len(cav1_box)):
        if cav1_box[i, 0, 0] < -100 or cav1_box[i, 0, 0] > 100 or cav1_box[i, 0, 1] < -40 or cav1_box[i, 0, 1] > 40:
            pass
        else:
            return_list.append(i)

    return return_list
