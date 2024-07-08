import torch
from data.Dataset import V2XSIMDataset, collate_fn
from data.config import Config, ConfigGlobal
from torch.utils.data import DataLoader
from utils.utils import get_gt_box

if __name__ == '__main__':
    config = Config('train', binary=True, only_det=True)
    config_global = ConfigGlobal('train', binary=True, only_det=True)
    trainset = V2XSIMDataset(dataset_roots=[f'../v2x-sim-1.0/train/agent{i}' for i in range(5)], config=config, config_global=config_global, split='train', adversarail_training=True)
    loader = DataLoader(trainset, batch_size=1, num_workers=8, collate_fn=collate_fn)
    # import ipdb;ipdb.set_trace()
    for i, sample in enumerate(loader):
        padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, gt_max_iou,\
                target_agent_id_list, num_agent_list, trans_matrices_list = zip(*sample)
        
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
        num_agent = torch.stack(tuple(num_agent_list), 1)

        padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        gt_bbox = []
        for i in range(len(num_agent)):
            num_sensor = num_agent[i][0].item()
            gt_bbox.append([])
            for k in range(num_sensor):
                reg_target_k = reg_target[i*5 + k].detach().cpu().numpy()
                anchors_map_k = anchors_map[i*5 + k].detach().cpu().numpy()
                gt_max_iou_idx = gt_max_iou[k][i]['gt_box']
                gt_bbox[-1].append(get_gt_box(anchors_map_k, reg_target_k, gt_max_iou_idx))
        break
    import ipdb;ipdb.set_trace()
