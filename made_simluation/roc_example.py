import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import sys
sys.path.append("/DB/data/yanghengzhao-1/adversarial/DAMC/yanghengzhao/disco-net")
from utils.bh_procedure import BH
sys.path.pop()

###############
# multi test
###############

# load score
with open("/DB/data/yanghengzhao-1/adversarial/DAMC/yanghengzhao/disco-net/experiments/multi-test-raev2/N01_E1e-01_S10/result.pkl", 'rb') as f:
    result = pickle.load(f)

score = np.concatenate([s for s in result['score'] if len(s) == 12])  # (N, 2) array
label = np.concatenate([l for l in result['label'] if len(l) == 12])  # (N,) array
pred = np.concatenate([p for p in result['pred'] if len(p) == 12])

dists = []
dists.append(np.load("/DB/data/yanghengzhao-1/adversarial/DAMC/yanghengzhao/disco-net/match_costs_validation.npy"))
dists.append(np.load("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/mreconstruction_loss_validation_v2.npy"))

def multi_test(dists, fdr, label, score):
    bh_test = BH(dists, fdr)
    is_attacker = []
    for s in score:
        rejected = bh_test.test(s)
        if len(rejected) > 0:
            is_attacker.append(1)
        else:
            is_attacker.append(0)
    is_attacker = np.array(is_attacker, dtype=np.int64)
    pred = is_attacker
    
    tp = ((label == 1) & (pred == 1)).sum()
    tn = ((label == 0) & (pred == 0)).sum()
    fp = ((label == 0) & (pred == 1)).sum()
    fn = ((label == 1) & (pred == 0)).sum()
    tpr = tp / (tp + fn + 1e-7)
    fpr = fp / (fp + tn + 1e-7)
    return tpr, fpr

tprs = []
fprs = []

for fdr in np.linspace(0, 2.2, 200):
    tpr, fpr = multi_test(dists, fdr, label, score)
    # print(f"fdr:{fdr}, tpr:{tpr}, fpr:{fpr}")
    tprs.append(tpr)
    fprs.append(fpr)
tprs = np.array(tprs)
fprs = np.array(fprs)

tprs = np.r_[tprs, 1.0]
fprs = np.r_[fprs, 1.0]

# plt.plot(fprs, tprs, label=f"auc={metrics.auc(fprs, tprs):.04f}")
# plt.xlabel("FPR",fontsize=15)
# plt.ylabel("TPR",fontsize=15)
# plt.show()

###############
# single test
###############

score = score[:, 0]  # (N,) array
fpr, tpr, threshold = metrics.roc_curve(label, score)
# plt.plot(fpr, tpr, label=f"auc={metrics.auc(fpr, tpr):.04f}")
# plt.xlabel("FPR",fontsize=15)
# plt.ylabel("TPR",fontsize=15)
# plt.show()