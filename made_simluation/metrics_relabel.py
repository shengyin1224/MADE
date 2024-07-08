import os
import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

"""
pgd:
    python metrics_relabel.py --relabel -t pgd --att_subpath N01_E1e-01_S10
ours:
    python metrics_relabel.py --relabel -t ours --att_subpath N01_E1e-1_S10_sep

"""
size = 30
new_size = 22
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel", action='store_true')
    parser.add_argument("-t", "--type", type=str)
    parser.add_argument("--att_subpath", type=str)
    args, _ = parser.parse_known_args()
    return args

def relabel_pgd(att_subpath):
    with open("/GPFS/data/shengyin/damc-yanghengzhao/disco-net/experiments/attack/single/N01_E1e-01_S10/result.pkl", 'rb') as f:
        eval_results1 = pickle.load(f)
    with open("experiments/gt/single/N01_E1e-01_S10/result.pkl", 'rb') as f:
        eval_results2 = pickle.load(f)

    ap1 = np.concatenate([np.array([r[f'agent_{i} mAP@0.5'] for i in range(4)]) for r in eval_results1 if len(r) == 8])
    ap2 = np.concatenate([np.array([r[f'agent_{i} mAP@0.5'] for i in range(4)]) for r in eval_results2 if len(r) == 8])

    for defense in ["match_cost_v2"]:
        with open(f"experiments/match_cost_v2/N01_E1e-01_S10/result.pkl", 'rb') as f:
            results = pickle.load(f)
        label = np.concatenate([l for l in results['label'] if len(l) == 12])
        pred = np.concatenate([p for p in results['pred'] if len(p) == 12])

        label[label == 1] = (ap1 < ap2 - 0.05)

        tp = ((label == 1) & (pred == 1)).sum()
        tn = ((label == 0) & (pred == 0)).sum()
        fp = ((label == 0) & (pred == 1)).sum()
        fn = ((label == 1) & (pred == 0)).sum()

        tpr = tp / (tp + fn + 1e-7)
        fpr = fp / (fp + tn + 1e-7)
        print(f"{defense}:\nTPR: {tpr:04f}, FPR:{fpr:04f}")

        if defense in ["match_cost_v2"]:
            # roc curve
            score = np.concatenate([s for s in results['score'] if len(s) == 12])
            with open(f"roc_curves/{defense}.pkl", 'wb') as f:
                pickle.dump({"score": score, "label": label}, f)

            fprs, tprs, thresholds = roc_curve(label, score)
            roc_auc = auc(fprs, tprs)
            np.save(f"roc_curves/{defense}_fprs.npy", fprs)
            np.save(f"roc_curves/{defense}_tprs.npy", tprs)
            np.save(f"roc_curves/{defense}_roc_auc.npy", roc_auc)

            plt.plot(fprs, tprs, label='ROC (AUC=%0.3f)' % roc_auc)
            plt.tick_params(labelsize=size)
            plt.legend(fontsize=new_size)
            plt.tight_layout()
            plt.savefig(f"roc_curves/{defense}.png")
            plt.clf()

def relabel_ours(att_subpath):
    with open("experiments/attack/shift_pred/N01_E1e-1_S10_sep/result.pkl", 'rb') as f:
        eval_results1 = pickle.load(f)
    with open("experiments/gt/shift_pred/N01_E1e-1_S10_sep/result.pkl", 'rb') as f:
        eval_results2 = pickle.load(f)

    ap1 = np.concatenate([np.array([r[f'agent_{i} mAP@0.5'] for i in range(4)]) for r in eval_results1 if len(r) == 8])
    ap2 = np.concatenate([np.array([r[f'agent_{i} mAP@0.5'] for i in range(4)]) for r in eval_results2 if len(r) == 8])

    for defense in ["raw_ae", "residual_ae_v2", "match_cost_v2", "multi-test-v3"]:
        with open(f"experiments/{defense}/{att_subpath}/result.pkl", 'rb') as f:
            results = pickle.load(f)
        label = np.concatenate([l for l in results['label'] if len(l) == 12])
        pred = np.concatenate([p for p in results['pred'] if len(p) == 12])

        label[label == 1] = (ap1 < ap2 - 0.05)

        tp = ((label == 1) & (pred == 1)).sum()
        tn = ((label == 0) & (pred == 0)).sum()
        fp = ((label == 0) & (pred == 1)).sum()
        fn = ((label == 1) & (pred == 0)).sum()

        tpr = tp / (tp + fn + 1e-7)
        fpr = fp / (fp + tn + 1e-7)
        print(f"{defense}:\nTPR: {tpr:04f}, FPR:{fpr:04f}")

        if defense in ["raw_ae", "residual_ae_v2", "match_cost_v2"]:
            # roc curve
            score = np.concatenate([s for s in results['score'] if len(s) == 12])
            fprs, tprs, thresholds = roc_curve(label, score)
            roc_auc = auc(fprs, tprs)

            plt.plot(fprs, tprs, label='ROC (AUC=%0.3f)' % roc_auc)
            plt.tick_params(labelsize=size)
            plt.legend(fontsize=new_size)
            plt.tight_layout()
            plt.savefig(f"roc_curves/{defense}.png")
            plt.close()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.relabel and args.type == 'pgd':
        relabel_pgd(args.att_subpath)
    elif args.relabel and args.type == 'ours':
        relabel_ours(args.att_subpath)
