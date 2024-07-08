from typing import List
import cv2
import torch
import numpy as np
from shapely.geometry import Polygon
from torchattacks.attack import Attack
from .pgd import PGD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .rm_com_pair import rm_com_pair
from data.obj_util import *

DILATE_KERNEL = np.ones((3, 3))

class ShiftBox(Attack):
    def __init__(self, model, n_att, attack_mode, colla_attack, attack_config_file, **kwargs) -> None:
        if hasattr(model, "module"):
            super().__init__("Shift", model.module)
        else:
            super().__init__("Shift", model)
        self.n_att = n_att
        self.attack_mode = attack_mode
        self.colla_attack = colla_attack
        self.attack_config_file = attack_config_file

        self.zero_vector = None

        self.at = kwargs.get("adversarial_training", False)
        self.random_erase_at = self.at
    
    @torch.no_grad()
    def forward(self, 
                bevs,
                trans_matrices, 
                num_agent_tensor, 
                batch_size, 
                anchors, 
                reg_loss_mask, 
                reg_targets, 
                labels,
                bboxes: List[List[np.ndarray]], # [[N x 4 x 2]]
                shift_directions: List[List[np.ndarray]]=None, # [[N x 2]]
                com_src=None, 
                com_tgt=None,):
        # import ipdb;ipdb.set_trace()
        bevs = bevs.clone().detach().to(self.device)
        trans_matrices = trans_matrices.clone().detach().to(self.device)
        voxels = bevs.detach().cpu().numpy().squeeze()
        maps = voxels.max(axis=-1)

        attack_srcs, attack_tgts = self.get_attack_pair(num_agent_tensor)

        bbox_corners = [[torch.from_numpy(b).float().to(self.device) for b in b_list] for b_list in bboxes]
        if shift_directions is not None:
            # FIXME check shift_directions
            shift_directions = [
                                [torch.from_numpy(s).long().to(self.device) 
                                if isinstance(s, np.ndarray)
                                else s.to(self.device) for s in s_list]
                                for s_list in shift_directions]
        else:
            shift_directions = self.get_shift_directions(bboxes, attack_tgts)
        assert sum(len(s) for s in shift_directions) == len(attack_srcs)
        # import ipdb;ipdb.set_trace()

        x_0, x_1, x_2, x_3, x_4 = self.model.encode(bevs)
        feat_maps, size = self.model.select_com_layer(x_0, x_1, x_2, x_3, x_4)

        attacks = []
        for i in range(len(attack_srcs)):
            trans = trans_matrices[attack_srcs[i, 0], attack_tgts[i, 1], attack_srcs[i, 1]].float()  # (4x4) 从 target坐标系变换到src坐标系
            rotation = trans[[0, 0, 1, 1], [0, 1, 0, 1]].reshape(2, 2)
            translation = trans[[0, 1], [3, 3]]
            translation[1] = -translation[1]

            ego_boxes = bbox_corners[attack_tgts[i, 0]][attack_tgts[i, 1]]
            target_boxes = ego_boxes.matmul(rotation) + translation
            shift_dir = shift_directions[attack_tgts[i, 0]][i]

            # 可视化验证
            """import ipdb;ipdb.set_trace()
            ego_map = maps[attack_tgts[i, 1]]
            adv_map = maps[attack_srcs[i, 1]]

            ego_boxes = ego_boxes.detach().cpu().numpy()
            target_boxes = target_boxes.detach().cpu().numpy()
            adv_boxes = bbox_corners[attack_srcs[i, 1]].detach().cpu().numpy()

            fig = plt.figure(1)
            m = np.zeros(ego_map.shape + (3,))
            m[ego_map == 0] = 255 * 0.99
            m[ego_map == 1] = [78, 52, 112]
            m = m.astype(np.uint8)
            plt.imshow(m, zorder=0)

            for corners in ego_boxes:
                corners = (corners - np.array([-32, -32])) / 0.25
                corners = np.concatenate([corners,corners[[0]]])
                plt.plot(corners[:,0], corners[:,1], c='g',linewidth=2,zorder=5)
            plt.xticks([])
            plt.yticks([])
            plt.savefig("ego_box.png", dpi=500)
            plt.close(1)

            fig = plt.figure(1)
            m = np.zeros(adv_map.shape + (3,))
            m[adv_map == 0] = 255 * 0.99
            m[adv_map == 1] = [78, 52, 112]
            m = m.astype(np.uint8)
            plt.imshow(m, zorder=0)

            for corners in target_boxes:
                corners = (corners - np.array([-32, -32])) / 0.25
                corners = np.concatenate([corners,corners[[0]]])
                plt.plot(corners[:,0], corners[:,1], c='g',linewidth=2,zorder=5)
            plt.xticks([])
            plt.yticks([])
            plt.savefig("target_box.png", dpi=500)
            plt.close(1)

            fig = plt.figure(1)
            m = np.zeros(adv_map.shape + (3,))
            m[adv_map == 0] = 255 * 0.99
            m[adv_map == 1] = [78, 52, 112]
            m = m.astype(np.uint8)
            plt.imshow(m, zorder=0)

            for corners in adv_boxes:
                corners = (corners - np.array([-32, -32])) / 0.25
                corners = np.concatenate([corners,corners[[0]]])
                plt.plot(corners[:,0], corners[:,1], c='g',linewidth=2,zorder=5)
            plt.xticks([])
            plt.yticks([])
            plt.savefig("adv_box.png", dpi=500)
            plt.close(1)
            import ipdb;ipdb.set_trace()
"""
        
            boxes_in_fm = (target_boxes + 32) / 0.25 / 256 * 32
            in_range = select_in_range_boxes(boxes_in_fm)
            boxes_in_range = boxes_in_fm[in_range]
            shift_dir = shift_dir[in_range]
            if self.random_erase_at and shift_dir.shape[0] > 0:
                # import ipdb;ipdb.set_trace()
                shift_dir[torch.rand(shift_dir.shape[0]) > 0.5] = -50  # random select box to erase


            feat_map = feat_maps[attack_srcs[i, 1]]
            if len(boxes_in_range) > 0:
                attacked_fm = self.shift_box(feat_map, boxes_in_range, shift_dir)
                # import ipdb;ipdb.set_trace()
                attacks.append(attacked_fm - feat_map)
            else:
                attacks.append(torch.zeros_like(feat_map))
        
        if len(attacks) > 0:
            attacks = torch.stack(attacks, dim=0)
        else:
            attacks = bevs.new_zeros((0,) + size[1:])
        return attacks

    def get_attack_pair(self, num_agent_tensor: torch.Tensor):
        mode = self.attack_mode
        n_att = self.n_att
        att_srcs = []
        att_tgts = []

        if mode == 'self':
            assert n_att == 1, "Number of attacker should be 1 if the attacker is the agent itself."
            att_src = []
            att_tgt = []
            for b, num_agent in enumerate(num_agent_tensor):
                for i, n in enumerate(num_agent):
                    if n > 0:
                        att_src.append([b, i])
                        att_tgt.append([b, i])
            att_src = torch.Tensor(att_src).long().to(num_agent_tensor.device)
            att_tgt = torch.Tensor(att_tgt).long().to(num_agent_tensor.device)
            # return att_src, att_tgt
            att_srcs.append(att_src)
            att_tgts.append(att_tgt)

        elif mode == 'others':
            if n_att > 1 and not self.colla_attack:
                for j in range(n_att):
                    att_src = []
                    att_tgt = []

                    for b, num_agent in enumerate(num_agent_tensor):
                        for i, n in enumerate(num_agent):
                            if n <= n_att:
                                continue
                            base = n.item()
                            assert (i+j+1) % base != i
                            att_src.append([b, (i+j+1) % base])
                            att_tgt.append([b, i])
                    att_src = torch.Tensor(att_src).long().to(num_agent_tensor.device)
                    att_tgt = torch.Tensor(att_tgt).long().to(num_agent_tensor.device)

                    att_srcs.append(att_src)
                    att_tgts.append(att_tgt)
                # return att_srcs, att_tgts
            
            else: # n_attack == 1 or n_attack > 1 and self.colla_attack
                att_src = []
                att_tgt = []
                for b, num_agent in enumerate(num_agent_tensor):
                    for i, n in enumerate(num_agent):
                        if n <= n_att:
                            continue
                        base = n.item()
                        for j in range(n_att):
                            assert (i+j+1) % base != i
                            att_src.append([b, (i+j+1) % base])
                            att_tgt.append([b, i])
                att_src = torch.Tensor(att_src).long().to(num_agent_tensor.device)
                att_tgt = torch.Tensor(att_tgt).long().to(num_agent_tensor.device)
                # return att_src, att_tgt
                att_srcs.append(att_src)
                att_tgts.append(att_tgt)
        else:
            raise NotImplementedError(mode)

        return torch.cat(att_srcs, dim=0), torch.cat(att_tgts, dim=0)

    def shift_box(self, 
            feat_map: torch.Tensor, # C x H x W
            target_boxes: torch.Tensor, # N x 4 x 2
            shift_directions: torch.Tensor, # N x 2 or None
        ):
        feat_map = feat_map.clone()
        # return torch.zeros_like(feat_map)
        # 
        yy, xx = torch.meshgrid(torch.arange(feat_map.shape[1]), 
                                torch.arange(feat_map.shape[2]))

        xx = xx.unsqueeze(-1).repeat(1, 1, target_boxes.shape[0]).float().to(feat_map.device)
        yy = yy.unsqueeze(-1).repeat(1, 1, target_boxes.shape[0]).float().to(feat_map.device)

        A, B, C, D = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        dist1 = torch.sign(
            (xx - A[:, 0]) * (B[:, 1] - A[:, 1]) - (yy - A[:, 1]) * (B[:, 0] - A[:, 0])
        )
        dist2 = torch.sign(
            (xx - B[:, 0]) * (C[:, 1] - B[:, 1]) - (yy - B[:, 1]) * (C[:, 0] - B[:, 0])
        )
        dist3 = torch.sign(
            (xx - C[:, 0]) * (D[:, 1] - C[:, 1]) - (yy - C[:, 1]) * (D[:, 0] - C[:, 0])
        )
        dist4 = torch.sign(
            (xx - D[:, 0]) * (A[:, 1] - D[:, 1]) - (yy - D[:, 1]) * (A[:, 0] - D[:, 0])
        )

        in_box = (dist1 > 0).logical_and(dist2 > 0).logical_and(dist3 > 0).logical_and(dist4 > 0)
        in_box = in_box.cpu().numpy().astype(np.float32)
        # import ipdb;ipdb.set_trace()
        
        in_box = cv2.dilate(in_box, DILATE_KERNEL, 1)
        if (in_box.ndim == 2):
            # import ipdb;ipdb.set_trace()
            in_box = in_box[..., np.newaxis]
        in_box = torch.Tensor(in_box).to(self.device)

        # feat_map_vis = feat_map.sum(dim=0).cpu().numpy()
        # plt.imshow(feat_map_vis)
        # plt.savefig("test_fm.png")
        # plt.close()
        # import ipdb;ipdb.set_trace()
        for i in range(in_box.shape[-1]):
            x_rm, y_rm = torch.where(in_box[:, :, i])
            
            if shift_directions is not None:
                dx, dy = shift_directions[i]
                box_feat = feat_map[:, x_rm, y_rm]
                x_new = x_rm + dx
                y_new = y_rm + dy
                keep_mask = (x_new >= 0) & (x_new < feat_map.shape[1]) & (y_new >= 0) & (y_new < feat_map.shape[2])
                x_new = x_new[keep_mask]
                y_new = y_new[keep_mask]
                box_feat = box_feat[:, keep_mask]

                if self.zero_vector is None:
                    vectors, counts = torch.unique(feat_map.reshape(feat_map.shape[0], -1), 
                                                    return_counts=True, dim=1)
                    self.zero_vector = vectors[:, counts==counts.max()].detach()
                feat_map[:, x_rm, y_rm] = self.zero_vector # erase
                feat_map[:, x_new, y_new] = box_feat
            else:
                feat_map[:, x_rm, y_rm] = 0 # erase only
        # feat_map_vis = feat_map.sum(dim=0).cpu().numpy()
        # plt.imshow(feat_map_vis)
        # plt.savefig("test_fm2.png")
        # plt.close()
        # import ipdb;ipdb.set_trace()
        return feat_map

    def get_shift_directions(self, bboxes, attack_tgt):
        shift_directions = [[] for _ in range(attack_tgt[:, 0].max() + 1)]
        if self.colla_attack:
            tmp_directions = [
                [
                    torch.randint(-1, 2, (len(b), 2), dtype=torch.long, device=self.device) 
                    for b in b_list
                ]
                for b_list in bboxes
            ]
            for tgt in attack_tgt:
                shift_directions[tgt[0]].append(tmp_directions[tgt[0]][tgt[1]])
        else:
            for tgt in attack_tgt:
                b = bboxes[tgt[0]][tgt[1]]
                shift_directions[tgt[0]].append(
                    torch.randint(-1, 2, (len(b), 2), dtype=torch.long, device=self.device)
                )
        return shift_directions
    
    def inference(self, bevs, attack, trans_matrices, num_agent_tensor, batch_size, com_src=None, com_tgt=None, random_smooth=False):   
        attack_src, attack_tgt = self.get_attack_pair(num_agent_tensor)
        outputs = self.model(bevs=bevs, trans_matrices=trans_matrices, batch_size=batch_size, num_agent_tensor=num_agent_tensor,
                com_src=com_src, com_tgt=com_tgt, attack=attack, attack_src=attack_src, attack_tgt=attack_tgt)
        return outputs

def select_in_range_boxes(boxes):
    """
    boxes: N x 4 x 2
    """
    in_range = (boxes[:, :, 0] >= 0) & (boxes[:, :, 0] < 32) & (boxes[:, :, 1] >= 0) & (boxes[:, :, 1] < 32)
    in_range = in_range.all(dim=1)
    return in_range

class ShiftBoxPGD(ShiftBox):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.shift = ShiftBox(**kwargs)
        self.pgd = PGD(**kwargs)
    
    def forward(self, 
                bevs,
                trans_matrices, 
                num_agent_tensor, 
                batch_size, 
                anchors, 
                reg_loss_mask, 
                reg_targets, 
                labels,
                bboxes: List[np.ndarray], # [N x 4 x 2]
                shift_directions: List[np.ndarray]=None, # [N x 2]
                com_src=None, 
                com_tgt=None,):
        attack1 = super().forward(
            bevs=bevs,
            trans_matrices=trans_matrices,
            num_agent_tensor=num_agent_tensor,
            batch_size=batch_size,
            anchors=anchors,
            reg_loss_mask=reg_loss_mask,
            reg_targets=reg_targets,
            labels=labels,
            bboxes=bboxes,
            shift_directions=shift_directions,
        )
        attack_src, attack_tgt = self.get_attack_pair(num_agent_tensor)

        attack2 = self.pgd.forward(
            bevs=bevs,
            trans_matrices=trans_matrices,
            num_agent_tensor=num_agent_tensor,
            batch_size=batch_size,
            anchors=anchors,
            reg_loss_mask=reg_loss_mask,
            reg_targets=reg_targets,
            labels=labels,
            com_src=com_src,
            com_tgt=com_tgt,
            ext_attack=attack1,
            ext_attack_src=attack_src,
            ext_attack_tgt=attack_tgt
        )
        return attack1 + attack2


class RegionBasedShiftBoxPGD(ShiftBox):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.shift = ShiftBox(**kwargs)
        self.pgd = PGD(**kwargs)
    
    def forward(self, 
                bevs,
                trans_matrices, 
                num_agent_tensor, 
                batch_size, 
                anchors, 
                reg_loss_mask, 
                reg_targets, 
                labels,
                bboxes: List[List[np.ndarray]], # [N x 4 x 2]
                shift_directions: List[np.ndarray]=None, # [N x 2]
                com_src=None, 
                com_tgt=None,):
        # 将bboxes分为范围内和范围外 FIXME 目前只支持一个attacker
        # step 1 没有attacker时，victim agent预测的box
        # import ipdb;ipdb.set_trace()
        attack_src, attack_tgt = self.get_attack_pair(num_agent_tensor)
        if attack_src.shape[0] == 0: # 没有attacker
            return bevs.new_zeros((0,) + self.model.com_size[1:])
        
        com_src_, com_tgt_ = self.model.get_default_com_pair(num_agent_tensor)
        com_src_, com_tgt_ = rm_com_pair(com_src_, com_tgt_, attack_src, attack_tgt)
        
        with torch.no_grad():
            victim_result, victim_box = self.model(bevs, trans_matrices, num_agent_tensor, 
                                    batch_size=batch_size,
                                    com_src=com_src_, com_tgt=com_tgt_,
                                    batch_anchors=anchors, nms=True)
        # import ipdb;ipdb.set_trace()
        # step 2 将bboxes根据上面预测的box分成两部分
        # only support batch_size = 1 FIXME
        vbbox_ref = [b[0]['pred'].squeeze(1) for b in victim_box]  # float32
        vbbox, ivbbox = self.split_bboxes(bboxes[0], vbbox_ref)
        # step 3 计算mask
        visible_mask = [regenerate_label(vbbox[i], ivbbox[i], anchors[i]) for i in range(len(vbbox))]
        while len(visible_mask) < len(anchors):
            visible_mask.append(np.zeros_like(visible_mask[0]))
        visible_mask = torch.Tensor(np.stack(visible_mask)).to(bevs.device)  # (N*5, 256, 256, 6)
        # visible_mask[visible_mask == 2] = 3

        # diff = (labels[..., 1] == 1) ^ (visible_mask != 0)
        # print(diff.sum(), 5 * 256 * 256 * 6)
        # visible_mask[diff] = 1
        # import ipdb;ipdb.set_trace()
        # same = (labels[reg_loss_mask[..., 0]][..., 0] == (visible_mask[reg_loss_mask[..., 0]] == 0))
        # print(same.sum(), -same.sum() + same.shape[0])
        # visible_mask = None
        # visible_mask = torch.ones_like(reg_loss_mask[..., 0]) 
        # visible_mask = torch.randint_like(reg_loss_mask[..., 0].float(), 1, 3) 
        # visible_mask = labels[..., 1].clone()
        # visible_mask[visible_mask == 1] = torch.randint_like(visible_mask[visible_mask == 1], 1, 3)

        # 可视化visible_mask
        # if os.path.exists("experiments/check_visible_mask"):
        #     base = "experiments/check_visible_mask"
        #     if not os.path.exists(os.path.join(base, f"{self.cnt:03d}")):
        #         os.mkdir(os.path.join(base, f"{self.cnt:03d}"))
        #     for i in range(num_agent_tensor[0][0]):
        #         for j in range(6):
        #             img = (visible_mask[i,:,:,j].cpu().numpy() * 125).astype(np.uint8)
        #             # import ipdb;ipdb.set_trace()
        #             cv2.imwrite(os.path.join(base, f"{self.cnt:03d}", f"{i}_{j}_mask.png"), 
        #                         img)
        #             img = (labels[i, :, :, j, 0].cpu().numpy() * 125).astype(np.uint8)
        #             cv2.imwrite(os.path.join(base, f"{self.cnt:03d}", f"{i}_{j}_label.png"), 
        #                         img)
        #     self.cnt += 1
        # import ipdb;ipdb.set_trace()
        if shift_directions is None:
            shift_directions = self.get_shift_directions(vbbox, ivbbox, attack_tgt)
        assert len(shift_directions) == len(attack_src)
        bboxes = [np.concatenate([vbbox[i], ivbbox[i]], axis=0) for i in range(len(vbbox))]


        # FIXME hack
        shift_directions = [shift_directions]
        bboxes = [bboxes]

        attack1 = super().forward(
            bevs=bevs,
            trans_matrices=trans_matrices,
            num_agent_tensor=num_agent_tensor,
            batch_size=batch_size,
            anchors=anchors,
            reg_loss_mask=reg_loss_mask,
            reg_targets=reg_targets,
            labels=labels,
            bboxes=bboxes,
            shift_directions=shift_directions,
        )
        

        attack2 = self.pgd.forward(
            bevs=bevs,
            trans_matrices=trans_matrices,
            num_agent_tensor=num_agent_tensor,
            batch_size=batch_size,
            anchors=anchors,
            reg_loss_mask=reg_loss_mask,
            reg_targets=reg_targets,
            labels=labels,
            com_src=com_src,
            com_tgt=com_tgt,
            ext_attack=attack1,
            ext_attack_src=attack_src,
            ext_attack_tgt=attack_tgt,
            visible_mask=visible_mask,
        )
        return attack1 + attack2

    def split_bboxes(self, bboxes, vbbox_ref):
        vbbox = []
        ivbbox = []
        for b, v in zip(bboxes, vbbox_ref):
            # 计算IoU
            v_polygons = [Polygon(vv) for vv in v]
            v_index = []
            for bb in b:
                bb_polygon = Polygon(bb)
                iou = max([bb_polygon.intersection(vv).area / bb_polygon.union(vv).area for vv in v_polygons])
                if iou > 0.5:
                    v_index.append(True)
                else:
                    v_index.append(False)
            v_index = np.array(v_index)
            vbbox.append(b[v_index])
            ivbbox.append(b[~v_index])
        return vbbox, ivbbox
    
    def get_shift_directions(self, vbboxes, ivbboxes, attack_tgt):
        shift_directions = []
        for tgt in attack_tgt:
            vb, ivb = vbboxes[tgt[1]], ivbboxes[tgt[1]]
            shift_directions.append(
                # erase <==> shift out of range 
                torch.cat([torch.randint(-1, 2, (len(vb), 2), dtype=torch.long, device=self.device),
                        #    torch.full((len(vb), 2), -50, dtype=torch.long, device=self.device),
                           torch.full((len(ivb), 2), -50, dtype=torch.long, device=self.device)])
            )
        return shift_directions

def regenerate_label(vbboxes, ivbboxes, anchor_map):
    # import ipdb;ipdb.set_trace()
    anchor_shape = anchor_map.shape[:3]
    anchor_map = anchor_map.cpu().numpy()
    anchor_corners = get_anchor_corners_list(anchor_map, 6)

    num_v = len(vbboxes)
    num_iv = len(ivbboxes)

    bboxes = np.concatenate([vbboxes, ivbboxes], axis=0)
    overlaps = compute_overlaps_gen_gt(anchor_corners, bboxes)

    association_map = (np.ones((overlaps.shape[0]))*(-1)).astype(np.int32)
    association_map[np.amax(overlaps,axis=1)>0.] = np.argmax(overlaps,axis=1)[np.max(overlaps,axis=1)>0]
    association_map = association_map.reshape(anchor_shape)

    visible_map = np.zeros_like(association_map)
    visible_map[(association_map>=0) & (association_map < num_v)] = 1
    visible_map[association_map >= num_v] = 2
    visible_map = visible_map.astype(np.int32)
    return visible_map  # 0: bg, 1: visible, 2: invisible

class EraseBox(ShiftBox):
    def get_shift_directions(self, bboxes, attack_tgt):
        shift_directions = [[] for _ in range(attack_tgt[:, 0].max() + 1)]
        if self.colla_attack:
            tmp_directions = [
                [
                    torch.full((len(b), 2), -50, dtype=torch.long, device=self.device) 
                    for b in b_list
                ]
                for b_list in bboxes
            ]
            for tgt in attack_tgt:
                shift_directions[tgt[0]].append(tmp_directions[tgt[0]][tgt[1]])
        else:
            for tgt in attack_tgt:
                b = bboxes[tgt[0]][tgt[1]]
                shift_directions[tgt[0]].append(
                    torch.full((len(b), 2), -50, dtype=torch.long, device=self.device)
                )
        return shift_directions

class EraseBoxPGD(EraseBox):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.shift = ShiftBox(**kwargs)
        self.pgd = PGD(**kwargs)
    
    def forward(self, 
                bevs,
                trans_matrices, 
                num_agent_tensor, 
                batch_size, 
                anchors, 
                reg_loss_mask, 
                reg_targets, 
                labels,
                bboxes: List[np.ndarray], # [N x 4 x 2]
                shift_directions: List[np.ndarray]=None, # [N x 2]
                com_src=None, 
                com_tgt=None,):
        attack1 = super().forward(
            bevs=bevs,
            trans_matrices=trans_matrices,
            num_agent_tensor=num_agent_tensor,
            batch_size=batch_size,
            anchors=anchors,
            reg_loss_mask=reg_loss_mask,
            reg_targets=reg_targets,
            labels=labels,
            bboxes=bboxes,
            shift_directions=shift_directions,
        )
        attack_src, attack_tgt = self.get_attack_pair(num_agent_tensor)

        attack2 = self.pgd.forward(
            bevs=bevs,
            trans_matrices=trans_matrices,
            num_agent_tensor=num_agent_tensor,
            batch_size=batch_size,
            anchors=anchors,
            reg_loss_mask=reg_loss_mask,
            reg_targets=reg_targets,
            labels=labels,
            com_src=com_src,
            com_tgt=com_tgt,
            ext_attack=attack1,
            ext_attack_src=attack_src,
            ext_attack_tgt=attack_tgt
        )
        return attack1 + attack2
