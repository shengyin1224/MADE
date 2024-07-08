import torch
import torch.nn.functional as F
import numpy as np
from .autoencoder import build_autoencoder
from .bh_procedure import build_bh_procedure

reconstruction_ae = build_autoencoder()
if torch.cuda.is_available():
    reconstruction_ae.cuda()

# bh_test = build_bh_procedure(dists = [np.random.rand(100), np.random.rand(100)])

bh_test = build_bh_procedure(dists = [
np.load("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/match_costs_validation.npy"),
np.load("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/mreconstruction_loss_validation.npy")])

@torch.no_grad()
def recon_loss(x):
    rec, _ = reconstruction_ae(x)
    loss = F.mse_loss(x, rec, reduction="none")
    loss = loss.sum(dim=[1, 2, 3])
    # return x.sum(dim=tuple(range(1, x.ndim))).detach().cpu().numpy()
    return loss.detach().cpu().numpy()

@torch.no_grad()
def attack_detection(
    model: torch.nn.Module,
    bev: torch.Tensor,
    trans_matrices: torch.Tensor,
    num_agent: torch.Tensor,
    anchors: torch.Tensor,
    attack: torch.Tensor,
    attack_src: torch.Tensor,
    attack_tgt: torch.Tensor,
    batch_size = 1,

):
    """
        return: 
            com_src, com_tgt
    """
    # encode feature
    x_0, x_1, x_2, x_3, x_4 = model.encode(bev)

    # select features to communicate and fuse
    feat_maps, size = model.select_com_layer(x_0, x_1, x_2, x_3, x_4)

    # com pairs for attack detection
    com_src_list, com_tgt_list = model.get_attack_det_com_pairs(num_agent)

    box_results = []  # 
    fused_features = []
    for com_src, com_tgt in zip(com_src_list, com_tgt_list):
        # fuse feature
        if attack is not None:
            perturb = model.place_attack_v2(
                attack, attack_src, attack_tgt, com_src, com_tgt)
            fused_feat = model.communication_attack(
                        feat_maps, trans_matrices, com_src, com_tgt, size, perturb, batch_size)
        else:
            fused_feat = model.communication_v2(
                feat_maps, trans_matrices, com_src, com_tgt, size, batch_size
            )
        fused_features.append(fused_feat[:num_agent[0, 0]])

        # final output
        x = model.decode(*model.place_merged_feature_back(x_0,
                                x_1, x_2, x_3, x_4, fused_feat), batch_size)
        result = model.head_out(x)
        box_result = model.post_process(result, anchors, num_agent[0, 0])
        box_results.append(box_result)

    ego_feature = fused_features[0]
    ego_box_result = box_results[0]

    per_agent_fused_features = fused_features[1:]
    per_agent_box_result = box_results[1:]

    match_costs = np.array([model.matcher(br, ego_box_result) for br in per_agent_box_result])
    reconstruction_loss = np.array([recon_loss(ff - ego_feature) for ff in per_agent_fused_features])
    
    if match_costs.ndim == 3:
        match_costs = match_costs[:, 0]

    src_id = np.zeros_like(match_costs, dtype=np.int32)
    tgt_id = np.zeros_like(match_costs, dtype=np.int32)

    # import ipdb;ipdb.set_trace()
    for i in range(num_agent[0, 0].item()):
        for j in range(num_agent[0, 0].item() - 1):
            tgt_id[j, i] = i
            offset = j + 1
            src_id[j, i] = (i + offset) % num_agent[0, 0].item()

    # flatten for multi-test
    match_costs = match_costs.flatten()
    reconstruction_loss = reconstruction_loss.flatten()
    scores_for_test = np.stack([match_costs, reconstruction_loss], axis=1)
    src_id = src_id.flatten()
    tgt_id = tgt_id.flatten()

    detected_index = []
    for i, score in enumerate(scores_for_test):
        rejected = bh_test.test(score)
        if len(rejected) > 0:
            detected_index.append(i)
    # print(detected_index)
    detected_src = src_id[detected_index]
    detected_tgt = tgt_id[detected_index]

    com_src, com_tgt = model.get_default_com_pair(num_agent)
    com_src, com_tgt = model.remove_com_pair(com_src, com_tgt, detected_src, detected_tgt)

    # return match_costs, reconstruction_loss
    return com_src, com_tgt