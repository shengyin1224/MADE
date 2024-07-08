import os
import numpy as np
import pickle
import argparse
from sklearn import metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logpath", type=str, help="log path")
    args = parser.parse_args()

    with open(os.path.join(args.logpath, "result.pkl"), 'rb') as f:
        result = pickle.load(f)
    score = np.concatenate([s for s in result['score'] if len(s) == 12])
    label = np.concatenate([l for l in result['label'] if len(l) == 12])
    pred = np.concatenate([p for p in result['pred'] if len(p) == 12])

    tp = ((label == 1) & (pred == 1)).sum()
    tn = ((label == 0) & (pred == 0)).sum()
    fp = ((label == 0) & (pred == 1)).sum()
    fn = ((label == 1) & (pred == 0)).sum()

    if score.ndim == 1:
        fpr, tpr, thresholds = metrics.roc_curve(label, score)
        roc_auc = metrics.auc(fpr, tpr)
    else:
        roc_auc = 0

    with open(os.path.join(args.logpath, "attack_detection_metrics.txt"), 'a') as f:
        f.write(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}\n")
        f.write(f"tpr: {tp / (tp + fn + 1e-7)}, fpr: {fp / (fp + tn + 1e-7)}\n")
        f.write(f"Precision: {tp / (tp + fp + 1e-7):.04f}\n")
        f.write(f"Recall: {tp / (tp + fn + 1e-7):.04f}\n")
        f.write(f"AUC: {roc_auc:.04f}\n")