import os
import pickle
import numpy as np

def relabel_one_attacker():
    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/match_cost_v2/N01_E1e-01_S10/result.pkl", 'rb') as f:
        mc_result = pickle.load(f)

    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/multi-test-raev2/N01_E1e-01_S10/result.pkl", 'rb') as f:
        mt_result = pickle.load(f)

    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/match_cost/N01_E1e-01_S10/AP05.pkl", 'rb') as f:
        aps = pickle.load(f)

    mc_pred = np.concatenate([p for p in mc_result['pred'] if len(p) == 12])
    mt_pred = np.concatenate([p for p in mt_result['pred'] if len(p) == 12])
    label = np.concatenate([l for l in mc_result['label'] if len(l) == 12])

    agent4_aps = aps['agent4_APs']
    all_aps = aps['all_APs']

    single_APs = []
    fused_APs = []
    for aps in agent4_aps:
        num_agent = len(aps)
        single_ap = np.array(aps[0])
        single_ap = np.concatenate([single_ap for _ in range(num_agent-1)])
        fused_ap = np.array(aps[1:])
        fused_ap = fused_ap.flatten()
        single_APs.append(single_ap)
        fused_APs.append(fused_ap)
    single_APs = np.concatenate(single_APs)
    fused_APs = np.concatenate(fused_APs)

    relabel = (single_APs - fused_APs > 0.05) & (label == 1)

    tp = ((relabel == 1) & (mc_pred == 1)).sum()
    tn = ((relabel == 0) & (mc_pred == 0)).sum()
    fp = ((relabel == 0) & (mc_pred == 1)).sum()
    fn = ((relabel == 1) & (mc_pred == 0)).sum()

    print("Match Cost:")
    print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    print(f"Precision: {tp / (tp + fp + 1e-7)}, Recall: {tp / (tp + fn + 1e-7)}")

    tp = ((relabel == 1) & (mt_pred == 1)).sum()
    tn = ((relabel == 0) & (mt_pred == 0)).sum()
    fp = ((relabel == 0) & (mt_pred == 1)).sum()
    fn = ((relabel == 1) & (mt_pred == 0)).sum()

    print("Multi-test:")
    print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    print(f"Precision: {tp / (tp + fp + 1e-7)}, Recall: {tp / (tp + fn + 1e-7)}")

def relabel_two_attacker():
    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/match_cost_v2/non-collaborative/N02_E1e-01_S10/result.pkl", 'rb') as f:
        mc_result = pickle.load(f)

    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/multi-test-raev2/non-collaborative/N02_E1e-01_S10/result.pkl", 'rb') as f:
        mt_result = pickle.load(f)

    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/save_11_bboxes/non-collaborative/N02_E1e-01_S10/AP05.pkl", 'rb') as f:
        aps = pickle.load(f)

    mc_pred = np.concatenate([p for p in mc_result['pred'] if len(p) == 12])
    mt_pred = np.concatenate([p for p in mt_result['pred'] if len(p) == 12])
    label = np.concatenate([l for l in mc_result['label'] if len(l) == 12])

    agent4_aps = aps['agent4_APs']
    all_aps = aps['all_APs']

    single_APs = []
    fused_APs = []
    for aps in agent4_aps:
        num_agent = len(aps)
        single_ap = np.array(aps[0])
        single_ap = np.concatenate([single_ap for _ in range(num_agent-1)])
        fused_ap = np.array(aps[1:])
        fused_ap = fused_ap.flatten()
        single_APs.append(single_ap)
        fused_APs.append(fused_ap)
    single_APs = np.concatenate(single_APs)
    fused_APs = np.concatenate(fused_APs)

    relabel = (single_APs - fused_APs > 0.05) & (label == 1)

    tp = ((relabel == 1) & (mc_pred == 1)).sum()
    tn = ((relabel == 0) & (mc_pred == 0)).sum()
    fp = ((relabel == 0) & (mc_pred == 1)).sum()
    fn = ((relabel == 1) & (mc_pred == 0)).sum()

    print("Match Cost:")
    print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    print(f"Precision: {tp / (tp + fp + 1e-7)}, Recall: {tp / (tp + fn + 1e-7)}")

    tp = ((relabel == 1) & (mt_pred == 1)).sum()
    tn = ((relabel == 0) & (mt_pred == 0)).sum()
    fp = ((relabel == 0) & (mt_pred == 1)).sum()
    fn = ((relabel == 1) & (mt_pred == 0)).sum()

    print("Multi-test:")
    print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    print(f"Precision: {tp / (tp + fp + 1e-7)}, Recall: {tp / (tp + fn + 1e-7)}")

def relabel_three_attacker():
    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/match_cost_v2/non-collaborative/N03_E1e-01_S10/result.pkl", 'rb') as f:
        mc_result = pickle.load(f)

    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/multi-test-raev2/non-collaborative/N03_E1e-01_S10/result.pkl", 'rb') as f:
        mt_result = pickle.load(f)

    with open("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/save_11_bboxes/non-collaborative/N03_E1e-01_S10/AP05.pkl", 'rb') as f:
        aps = pickle.load(f)

    mc_pred = np.concatenate([p for p in mc_result['pred'] if len(p) == 12])
    mt_pred = np.concatenate([p for p in mt_result['pred'] if len(p) == 12])
    label = np.concatenate([l for l in mc_result['label'] if len(l) == 12])

    agent4_aps = aps['agent4_APs']
    all_aps = aps['all_APs']

    single_APs = []
    fused_APs = []
    for aps in agent4_aps:
        num_agent = len(aps)
        single_ap = np.array(aps[0])
        single_ap = np.concatenate([single_ap for _ in range(num_agent-1)])
        fused_ap = np.array(aps[1:])
        fused_ap = fused_ap.flatten()
        single_APs.append(single_ap)
        fused_APs.append(fused_ap)
    single_APs = np.concatenate(single_APs)
    fused_APs = np.concatenate(fused_APs)

    relabel = (single_APs - fused_APs > 0.05) & (label == 1)

    tp = ((relabel == 1) & (mc_pred == 1)).sum()
    tn = ((relabel == 0) & (mc_pred == 0)).sum()
    fp = ((relabel == 0) & (mc_pred == 1)).sum()
    fn = ((relabel == 1) & (mc_pred == 0)).sum()

    print("Match Cost:")
    print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    print(f"Precision: {tp / (tp + fp + 1e-7)}, Recall: {tp / (tp + fn + 1e-7)}")

    tp = ((relabel == 1) & (mt_pred == 1)).sum()
    tn = ((relabel == 0) & (mt_pred == 0)).sum()
    fp = ((relabel == 0) & (mt_pred == 1)).sum()
    fn = ((relabel == 1) & (mt_pred == 0)).sum()

    print("Multi-test:")
    print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    print(f"Precision: {tp / (tp + fp + 1e-7)}, Recall: {tp / (tp + fn + 1e-7)}")

if __name__ == "__main__":
    # relabel_one_attacker()
    relabel_two_attacker()
    relabel_three_attacker()