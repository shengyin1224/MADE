import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.bh_procedure import build_bh_procedure

def load_dir(path):
    file_names = os.listdir(path)
    # samples = len(file_names)
    samples = 1789

    all_scores = []
    for i in range(samples):
        tmp = np.load(os.path.join(path, f"sample_{i}.npy"), allow_pickle=True)
        if tmp.shape[0] > 0:
            all_scores.append(
                np.array(tmp[0][0][0])
            )
        # import ipdb;ipdb.set_trace()
    
    return np.array(all_scores)



def dair_v2x():
    ml_cal = np.load("/GPFS/data/shengyin/OpencoodV2-Main/OpenCOODv2/generate_match/validation_0527/validation_match_cost.npy")
    crl_cal = np.load("/GPFS/data/shengyin/OpencoodV2-Main/OpenCOODv2/opencood/defense_model/calibration_outcome/validation_autoencoder.npy")

    ml_att = load_dir("/GPFS/data/shengyin/OpencoodV2-Main/OpenCOODv2/generate_match/test_set_/save_attack/0603_erase_and_shift_and_pgd_single_agent_prederase_and_shift_and_pgd_pred_single_agent_maxbbox_4length_eps5_iou0.5")
    crl_att = load_dir("/GPFS/data/shengyin/OpencoodV2-Main/OpenCOODv2/generate_ae_loss/residual/test_set_/save_attack/0603_erase_and_shift_and_pgd_single_agent_prederase_and_shift_and_pgd_pred_single_agent_maxbbox_4length_eps5_iou0.5")

    ml_norm = load_dir("/GPFS/data/shengyin/OpencoodV2-Main/OpenCOODv2/generate_match/test_set_single_agent/fuse_without_attack")
    crl_norm = load_dir("/GPFS/data/shengyin/OpencoodV2-Main/OpenCOODv2/generate_ae_loss/residual/test_set_single_agent/fuse_without_attack")

    dists = [ml_cal, crl_cal]
    bh = build_bh_procedure(dists, fdr=0.05)
    label = np.concatenate([np.zeros_like(ml_norm, dtype=np.int32), np.ones_like(ml_att, dtype=np.int32)])

    pred = []
    normal_pvs = []
    for x in zip(ml_norm, crl_norm):
        t = bh.test_v2(x)
        pv = bh.pv_conformal
        normal_pvs.append(pv)
        pred.append(t)
    normal_pvs = np.stack(normal_pvs)

    att_pvs = []
    for x in zip(ml_att, crl_att):
        t = bh.test_v2(x)
        pv = bh.pv_conformal
        att_pvs.append(pv)
        pred.append(t)
    att_pvs = np.stack(att_pvs)
    # import ipdb;ipdb.set_trace()
    pred = np.array(pred)

    tp = ((label == 1) & (pred == 1)).sum()
    tn = ((label == 0) & (pred == 0)).sum()
    fp = ((label == 0) & (pred == 1)).sum()
    fn = ((label == 1) & (pred == 0)).sum()

    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    print(f"Precision: {tp / (tp + fp + 1e-7):.04f}")
    print(f"Recall: {tp / (tp + fn + 1e-7):.04f}")
    print(f"FPR: {fp / (tn + fp + 1e-7):.04f}")

    plt.scatter(normal_pvs[:, 0], normal_pvs[:, 1], s=1, label='normal')
    plt.scatter(att_pvs[:, 0], att_pvs[:, 1], s=1, label='att')
    plt.axvline(0.05)
    plt.axhline(0.05)
    plt.legend()
    plt.title("dair v2x")
    plt.savefig('dair_bh.png')


if __name__ == "__main__":
    dair_v2x()
