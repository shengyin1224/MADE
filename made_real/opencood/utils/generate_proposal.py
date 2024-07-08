import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops import diff_iou_rotated_3d
from collections import OrderedDict
from torchattacks.attack import Attack
from opencood.utils import eval_utils
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
import matplotlib.pyplot as plt
from opencood.utils.box_utils import corner_to_center, corner_to_center_torch

def generate_erase_proposal(attack_target, real_data_dict, anchors, dataset, erase_fail_index, pred_gt_box_tensor = None, num = 0):

    # 生成erase_label

    tmp_erase_index = erase_fail_index
    if attack_target == 'gt':
        object_bbx_center = real_data_dict['object_bbx_center']
        object_bbx_mask = real_data_dict['object_bbx_mask']
        object_bbx_center = object_bbx_center[object_bbx_mask == 1]
    else:
        # attacked_bbox_center = corner_to_center(corner3d=pred_gt_box_tensor.cpu().detach().numpy(), order='hwl')
        # object_bbx_center = torch.tensor(attacked_bbox_center).cuda()
        object_bbx_center = corner_to_center_torch(corner3d=pred_gt_box_tensor, order='hwl')

    shift_index = []
    for j in range(object_bbx_center.shape[0]):
        if j not in tmp_erase_index:
            shift_index.append(j)

    tmp_erase_center = object_bbx_center[tmp_erase_index]
    tmp_shift_center = object_bbx_center[shift_index]
    erase_center = torch.zeros(size=(100,7))
    erase_center[:len(tmp_erase_index)] = tmp_erase_center

    erase_mask = torch.zeros(size=(100,))
    erase_mask[:len(tmp_erase_index)] = 1

    tmp_anchors = anchors.clone()
    tmp_anchors = tmp_anchors.cpu().detach().numpy()

    erase_mask = erase_mask.numpy()

    erase_center = erase_center.numpy()

    label_erase = dataset.post_processor.generate_label(gt_box_center=erase_center,
        anchors=tmp_anchors,
        mask=erase_mask)

    label_erase = torch.tensor(label_erase['pos_equal_one']).cuda()

    # 生成erase_proposal
    if attack_target == 'gt':
        label_erase = label_erase.squeeze(0).view(-1)    # (H*W, )
        fg_proposal_erase = label_erase == 1              # (H*W, )
    else:
        label_erase = label_erase.squeeze(0).view(-1, )
        label_erase = torch.sigmoid(label_erase)
        fg_proposal_erase = label_erase > 0.5             # (H*W, )

    fg_proposal_erase = fg_proposal_erase.cpu().detach().numpy()
    np.save(f'/GPFS/data/shengyin/OpenCOOD-main/outcome/erase_fail_index/proposal_erase/sample{num}.npy', fg_proposal_erase)
    return fg_proposal_erase
 