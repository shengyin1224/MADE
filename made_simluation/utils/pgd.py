# modified from from torchattacks
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks.attack import Attack

from utils.detection_util import bev_box_decode_torch, center_to_corner_box2d_torch, rescale_boxes
import matplotlib.pyplot as plt

import os 
import sys 
# sys.path.append('/DB/data/yanghengzhao/adversarial/Rotated_IoU')
# from oriented_iou_loss import cal_diou, cal_giou, cal_iou
from .iou_utils import cal_iou

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, 
                eps=0.1, alpha=0.1, steps=40, 
                attack_mode='others',
                n_att=1, colla_attack=True, 
                noise_attack=False,
                random_start=True,
                project=True,
                keep_sparsity=False,
                keep_pos=False,
                attack_classifier=False,
                vis_path=None,
                attack_config_file=None,
                **kwargs):
        if hasattr(model, "module"):
            super().__init__("PGD", model.module)
        else:
            super().__init__("PGD", model)
        """
            需要支持的功能：
            1. 设置攻击的超参数(eps, alpha, steps, project) -- done
            2. 设置attacker的src和tgt -- done
            3. 设置attacker个数 -- done
            4. targeted 攻击？ -- to do
            5. agent 之间的迁移 -- done 
            6. random smooth -- done
            7. multi-agent attack 是否合作 -- done
            8. 保持perturbed feature map的稀疏性 -- done
            9. 攻击attacker classifier
        """
        self.eps = eps
        self.alpha = alpha  
        self.gamma = 1
        self.steps = steps
        self.random_start = random_start
        self.attack_mode = attack_mode
        self.noise_attack = noise_attack
        self.project = project
        self.keep_sparsity = keep_sparsity
        self.keep_pos = keep_pos
        self.attack_classifier = attack_classifier

        self.n_att = n_att
        self.colla_attack = colla_attack
        self.attack_config_file = attack_config_file

        self.visualize = vis_path is not None
        if self.visualize:
            self.cnt = 0
            self.vis_path = vis_path
            if not os.path.exists(self.vis_path):
                os.makedirs(self.vis_path)
        
        if self.attack_classifier:
            from utils.binary_classifier import my_CNN as BinaryClassifier
            binary_classifier = BinaryClassifier("/DB/data/shengyin/classifer_of_adverisial/parameter_in_epoch_30.pth").cuda()
            binary_classifier.eval()
            self.attacker_classifier = binary_classifier

        # TODO implement targeted attack
        self._supported_mode = ['default']  # , 'targeted']

        # for save loss curve
        self.cnt = 0

        # for match distribution
        # self.normal_var_mean = torch.load("/DB/data/yanghengzhao/adversarial/disco_features/test/normal_var_mean.pth")
        # self.perturbed_var_mean = torch.load("/DB/data/yanghengzhao/adversarial/disco_features/test/N01_E5e-01_S10_sparse_var_mean.pth")
    @torch.enable_grad()
    def forward(self, 
                bevs, 
                trans_matrices, 
                num_agent_tensor, 
                batch_size, 
                anchors, 
                reg_loss_mask, 
                reg_targets, 
                labels,
                com_src=None, 
                com_tgt=None,
                ext_attack=None,
                ext_attack_src=None,
                ext_attack_tgt=None,
                visible_mask=None,):
        r"""
        Overridden.
        """
        bevs = bevs.clone().detach().to(self.device)
        trans_matrices = trans_matrices.clone().detach().to(self.device)

        anchors = anchors.clone().detach().to(self.device)
        reg_loss_mask = reg_loss_mask.clone().detach().to(self.device)
        reg_targets = reg_targets.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if com_src is None or com_tgt is None:
            com_src, com_tgt = self.model.get_default_com_pair(num_agent_tensor)
        # if self._targeted:
        #     target_labels = self._get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()
        # TODO
        # 1. 把正常模型输出的框作为gt，产生regression target和label (hard) done
        # 2. 把正常模型的每个anchor预测作为regression target和label (easy) done
        # 3. adam optimizer
        # 4. support broadcast attack

        attack_srcs, attack_tgts = self.get_attack_pair(num_agent_tensor, for_inf=False)

        com_size = self.model.com_size

        tmp = self.model.kd_flag
        self.model.kd_flag = False

        attacks = []
        for attack_src, attack_tgt in zip(attack_srcs, attack_tgts):
            attack = bevs.new_zeros(com_size).repeat(attack_src.shape[0], *[1 for _ in com_size[1:]])
            if attack.shape[0] == 0:
                attacks.append(attack)
                continue
            
            if self.random_start:
                # Starting at a uniformly random point
                attack.uniform_(-self.eps, self.eps)

            if self.noise_attack: 
                for _ in range(self.steps):
                    attack += self.alpha * torch.randn_like(attack) * 3
                    if self.project:
                        attack.clamp_(min=-self.eps, max=self.eps)
            else:
                feat_maps = self.model.encode(bevs)
                com_features = feat_maps[self.model.layer]
                zero_mask = (com_features != 0).float()
                zero_mask = zero_mask[attack_src[:, 1]]  # align with attack, only support batch_size == 1
                attack *= zero_mask

                losses = []
                for _ in range(self.steps):
                    attack.requires_grad = True
                    
                    if ext_attack is not None:
                        input_attack = self.model.place_attack_v2(ext_attack, ext_attack_src, ext_attack_tgt, attack_src, attack_tgt)
                    else:
                        input_attack = torch.zeros_like(attack)
                    if self.keep_sparsity:
                        outputs = self.model.after_encode(feat_maps, trans_matrices, com_src, com_tgt, 
                                                        attack * zero_mask + input_attack, attack_src, attack_tgt, batch_size)
                    else:
                        outputs = self.model.after_encode(feat_maps, trans_matrices, com_src, com_tgt, 
                                                        attack + input_attack, attack_src, attack_tgt, batch_size)

                    if not isinstance(outputs, dict):  # for KD
                        outputs = outputs[0]
                    # Calculate loss
                    if self._targeted:
                        # cost = -loss(outputs, target_labels)
                        raise NotImplementedError()
                    else:
                        cost = self.loss(outputs, anchors, reg_loss_mask, reg_targets, labels, visible_mask)
                        if self.attack_classifier:
                            classify_labels = torch.ones(attack.shape[0], dtype=torch.long, device=self.device)
                            perturbed_features = attack + com_features[attack_src[:, 1]]
                            cls_loss = F.nll_loss(torch.log(self.attacker_classifier(perturbed_features)), classify_labels)
                            cost += cls_loss

                    losses.append(cost.item())
                    # Update adversarial images
                    grad = torch.autograd.grad(cost, attack,
                                            retain_graph=False, create_graph=False)[0]
                    grad = grad.nan_to_num()

                    if self.keep_sparsity:
                        grad *= zero_mask

                    attack = attack.detach() - self.alpha * grad.sign()
                    # attack *= zero_mask
                    # print(f"step {_}: grad max {grad.max()}, grad nrom {grad.norm()}")
                    # attack = attack.detach() - self.alpha * grad / \
                    #     (torch.norm(grad.reshape(grad.shape[0], -1), dim=1).reshape(attack.shape[0], 1, 1, 1) + 1e-10)

                    if self.project:
                        attack.clamp_(min=-self.eps, max=self.eps)  # L infinity
                    
                    if self.keep_pos:
                        perturbed_features = attack + com_features[attack_src[:, 1]]
                        neg_index = perturbed_features < 0
                        attack[neg_index] -= perturbed_features[neg_index].detach() - 1e-7
                        # perturbed_features = attack + com_features[attack_src[:, 1]]
                        # if perturbed_features.min() < 0:
                        #     import ipdb;ipdb.set_trace() 
                        # if torch.norm(attack) > self.eps:
                        #     attack = attack * self.eps / torch.norm(attack)  # L2

                        # attack_norms = torch.norm(attack.view(batch_size, -1), p=2, dim=1)
                        # factor = self.eps / attack_norms
                        # factor = torch.min(factor, torch.ones_like(attack_norms))
                        # attack = attack * factor.view(-1, 1, 1, 1)                       # L2
                # torch.save(losses, f"/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/attack_loss/N01_E5e-01_S100_sparse_max/losses_{self.cnt}.pt")
                self.cnt += 1 

                # after optimization constraints
                # sparsity
                # attack = attack * zero_mask

                # positive
                # perturbed_features = attack + com_features[attack_src[:, 1]]
                # neg_index = perturbed_features < 0
                # attack[neg_index] -= perturbed_features[neg_index].detach() - 1e-7
                # perturbed_features = attack + com_features[attack_src[:, 1]]

                # match distribution
                # perturbed_features = attack + com_features[attack_src[:, 1]]
                # perturbed_features = (perturbed_features - self.perturbed_var_mean['mean'].cuda()) / self.perturbed_var_mean['var'].cuda() * self.normal_var_mean['var'].cuda() + self.normal_var_mean["mean"].cuda()
                # attack = perturbed_features - com_features[attack_src[:, 1]]

            attacks.append(attack)
        attacks = torch.cat(attacks, dim=0)

        # visualize
        if self.visualize:
            # FIXME 不兼容
            # only support eva_num=1
            num_agent = num_agent_tensor[0, 0]
            normal = self.model.encode(bevs)[3]
            placed_attack = torch.stack([self.model.place_attack(attack[:, i, ...], num_agent_tensor[:, i], target_index=i, mode=self.attack_mode) for i in range(num_agent)], dim=0)
            placed_attack = placed_attack.sum(dim=0)
            normal4viz = normal[:num_agent].detach().cpu().numpy()
            attack4viz = (placed_attack + normal)[:num_agent].detach().cpu().numpy()
            normal4viz = normal4viz.sum(axis=1)  # (num_agent, h, w)
            attack4viz = attack4viz.sum(axis=1)

            fig, axes = plt.subplots(2, num_agent.item())
            for i in range(num_agent):
                axes[0, i].set_title(f"Agent {i}, normal")
                axes[0, i].set_axis_off()
                # ax[0, i].imshow((normal4viz[i] - normal4viz[i].min()) / (normal4viz[i].max() - normal4viz[i].min()))
                im = axes[0, i].imshow(normal4viz[i])

                axes[1, i].set_title(f"Agent {i}, attacked")
                axes[1, i].set_axis_off()
                # ax[1, i].imshow((attack4viz[i] - attack4viz[i].min()) / (attack4viz[i].max() - attack4viz[i].min()))
                im = axes[1, i].imshow(attack4viz[i].clip(normal4viz[i].min(), normal4viz[i].max()))
            
            fig.colorbar(im, ax=axes, location='right', shrink=0.5, fraction=0.1)
            fig.savefig(f"{self.vis_path}/{self.cnt}.png")
            self.cnt += 1
            # import ipdb;ipdb.set_trace()
        self.model.kd_flag = tmp
        return attacks

    def inference(self, bevs, attack, trans_matrices, num_agent_tensor, batch_size, com_src=None, com_tgt=None, random_smooth=False):        
        attack_src, attack_tgt = self.get_attack_pair(num_agent_tensor)
        # random_smooth = True
        # TODO use random_smooth in outer code
        if random_smooth:
            attack += (torch.rand_like(attack) - 0.5) * 2 * self.alpha * self.steps * 2 # uniform noise 
            # attack += torch.randn_like(attack) * self.alpha * self.steps * 3  # gaussian noise
        outputs = self.model(bevs=bevs, trans_matrices=trans_matrices, batch_size=batch_size, num_agent_tensor=num_agent_tensor,
                com_src=com_src, com_tgt=com_tgt, attack=attack, attack_src=attack_src, attack_tgt=attack_tgt)
        return outputs

    def get_attack_pair0(self, num_agent_tensor: torch.Tensor, 
                                n_com: Optional[int] = None):
        mode = self.attack_mode
        n_att = self.n_att
        att_src = []
        att_tgt = []
        for b, num_agent in enumerate(num_agent_tensor):
            for i, n in enumerate(num_agent):
                if mode == 'others' and n <= n_att:
                    continue
                base = min(n.item(), n_com) if n_com is not None else n.item()
                if base <= 0:
                    continue
                for j in range(n_att):
                    if mode == 'others':
                        att_src.append([b, (i+j+1) % base])
                        att_tgt.append([b, i])
                    elif mode == 'self':
                        assert n_att == 1
                        att_src.append([b, i])
                        att_tgt.append([b, i])

        att_src = torch.Tensor(att_src).long().to(num_agent_tensor.device)
        att_tgt = torch.Tensor(att_tgt).long().to(num_agent_tensor.device)
        return att_src, att_tgt
    
    def get_attack_pair(self, num_agent_tensor: torch.Tensor, for_inf=True, 
                                n_com: Optional[int] = None):
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

        if for_inf:
            return torch.cat(att_srcs, dim=0), torch.cat(att_tgts, dim=0)
        else: # for pertubation generation
            return att_srcs, att_tgts

    def attack_transfer(self, attack, num_agent_tensor, att_src, att_tgt):
        # only support batch_size=1
        assert num_agent_tensor.shape[0] == 1
        num_agent = num_agent_tensor[0, 0].item()
        broadcast_attack = attack[0:1].repeat(num_agent-1, *[1 for _ in attack.shape[1:]])
        broadcast_src = att_src[0:1].repeat(num_agent-1, *[1 for _ in att_src.shape[1:]])
        broadcast_tgt = broadcast_src.new_tensor([[0, i] for i in range(num_agent) if i != att_src[0, 1].item() and i != att_tgt[0, 1].item()])

        return broadcast_attack, broadcast_src, broadcast_tgt

    def loss(self,
            result, # 'cls': (b*n, 256*256*6, 2), 'loc': (b*n, 256, 256, 6, 1, 6)
            anchors, # (b*n, 256, 256, 6, 6)
            reg_loss_mask, # (b*n, 256, 256, 6, 1) cls和loc都应该用
            reg_targets, # (b*n, 256, 256, 6, 1, 6) 正常模型的输出
            labels, # onehot encoding (b*n, 256*256*6, 2) 正常模型的输出
            visible_mask=None,
            ):
        
        pred_cls = result['cls']
        pred_loc = result['loc']

        labels = labels.reshape(labels.shape[0], -1, labels.shape[-1])
        
        # mask out padded agents
        pred_cls = F.softmax(pred_cls[reg_loss_mask.reshape(reg_loss_mask.shape[0], -1)], dim=-1) # (M, 2)
        pred_loc = pred_loc[reg_loss_mask] # (M, 6)
        anchors = anchors[reg_loss_mask.squeeze(-1)] # (M, 6)
        labels = labels[reg_loss_mask.reshape(reg_loss_mask.shape[0], -1)] # (M, 2)
        reg_targets = reg_targets[reg_loss_mask] # (M, 6)

        # select proposals by cls scores
        scores = F.softmax(labels, dim=-1)
        fg_proposal = scores[:, 1] > 0.7  #  (M, )

        # import ipdb;ipdb.set_trace()
        if visible_mask is not None:
            if visible_mask.dim() == 4:
                visible_mask = visible_mask.unsqueeze(-1)
            visible_mask = visible_mask[reg_loss_mask] # (M,)
        else:
            visible_mask = torch.ones_like(fg_proposal)

        vfg_proposal = fg_proposal & (visible_mask == 1)
        ivfg_proposal = fg_proposal & (visible_mask == 2)
        # to corner
        if vfg_proposal.any():
            decoded_pred = bev_box_decode_torch(pred_loc[vfg_proposal], anchors[vfg_proposal]) # (N, 6) [x, y, w, h, sin, cos]
            decoded_pred = rescale_boxes(decoded_pred)
            decoded_target = bev_box_decode_torch(reg_targets[vfg_proposal], anchors[vfg_proposal]) # (N, 6)
            decoded_target = rescale_boxes(decoded_target)

            def sincos2angle(x: torch.Tensor):
                # torch.atan2(sin, cos)
                return torch.atan2(x[..., 0], x[..., 1]).unsqueeze(-1)
            
            pred = torch.cat([decoded_pred[..., :4], sincos2angle(decoded_pred[..., 4:])], dim=-1)
            target = torch.cat([decoded_target[..., :4], sincos2angle(decoded_target[..., 4:])], dim=-1)
            # pred_corners = center_to_corner_box2d_torch(decoded_pred[..., :2], decoded_pred[..., 2:4], decoded_pred[..., 4:]) # (N, 4, 2)
            # target_corners = center_to_corner_box2d_torch(decoded_target[..., :2], decoded_target[..., 2:4], decoded_target[..., 4:]) # (N, 4, 2)

            # compute IoU
            # input [x, y, w, h, angle]
            iou = cal_iou(pred.unsqueeze(0), target.unsqueeze(0))[0].squeeze(0)
        else:
            iou = 0
        # loss 
        bg_proposal = scores[:, 0] > 0.3
        lamb = 0.2
        # lamb = 1.0
        # total_loss = torch.sum(- torch.log(1 - pred_cls[fg_proposal, 1]) * iou) + lamb * torch.sum(- pred_cls[bg_proposal, 0].pow(self.gamma) * torch.log(1 - pred_cls[bg_proposal, 0]))
        bg_loss = torch.sum(- pred_cls[bg_proposal, 0].pow(self.gamma) * torch.log(1 - pred_cls[bg_proposal, 0]))

        fg_loss = torch.sum(- torch.log(1 - pred_cls[vfg_proposal, 1]) * iou) + torch.sum(- torch.log(1 - pred_cls[ivfg_proposal, 1]))

        total_loss = lamb * bg_loss + fg_loss

        # import ipdb;ipdb.set_trace()
        return total_loss


if __name__ == "__main__":
    pgd = PGD(torch.nn.Conv2d(1,1,1,1).cuda(), 
                eps=0.1, alpha=0.1, steps=40, 
                attack_mode='others',
                n_att=1, colla_attack=False, 
                noise_attack=False,
                random_start=True,
                project=True,
                vis_path=None)
    for num_agent_tensor in [torch.Tensor([[2,2,0,0,0]]).cuda(), 
                             torch.Tensor([[3,3,3,0,0]]).cuda(),
                             torch.Tensor([[4,4,4,4,0]]).cuda()]:
        src, tgt = pgd.get_attack_pair(num_agent_tensor)
        print(num_agent_tensor)
        print("Src:")
        print(src)
        print("Tgt:")
        print(tgt)
        print("=====================")
