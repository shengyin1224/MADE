import pickle 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
sys.path.append('/dssg/home/acct-umjpyb/umjpyb/shengyin')
from opencood.utils.generate_npy import generate_npy

size = 30
new_size = 22
attack_type = 'erase_and_shift_and_pgd'
folder_path = 'opencood/roc_curve'

def draw_roc(attack_type, method, ae_type = 'residual', binary = False):
    """
        For raw_ae, residual_ae and match_cost
    """

    attack_target = 'pred'
    total_num = 1789
    save_path = 'outcome/score_and_label/' + attack_type

    if attack_type == 'pgd':
        save_dir1 = 'test_set_save_attack/1016_PGD_predpred_single_agent__1016_PGD_loss_1.5'
    else:
        save_dir1 = 'test_set_save_attack/1016_erase_and_shift_and_pgd__prederase_and_shift_and_pgd_pred_single_agent_maxbbox_4length_eps1.5_iou0.5'

    if method == 'match':
        defense_path = 'generate_match/'
        tmp_method = method
    elif method == 'ae' and ae_type == 'residual':
        defense_path = 'generate_ae_loss/residual/'
        tmp_method = method + '_' + ae_type
    else:
        defense_path = 'generate_ae_loss/raw/'
        tmp_method = method + '_' + ae_type

    save_dir2 = 'test_set_single_agent/fuse_without_attack'

    # 生成攻击对应的score, label
    if os.path.exists(save_path + 'score1.npy'):
        path1 = defense_path + save_dir1  
    score1, label1 = generate_npy(path=path1,save_path=save_path,method=method,attack_gt=attack_target)

    # 生成不攻击对应的score, label
    path2 = defense_path + save_dir2  
    score2, label2 = generate_npy(path=path2,save_path=save_path,method=method,attack_gt=attack_target)

    # 合在一起
    score = np.concatenate((score1, score2))
    label = np.concatenate((label1, label2))

    fpr, tpr, thresholds = roc_curve(label, score)
    plt.plot(fpr, tpr, label='ROC (AUC=%0.3f)' % auc(fpr, tpr))
    plt.tick_params(labelsize=size)
    plt.legend(fontsize=new_size)
    plt.tight_layout()
    if not binary:
        plt.savefig(folder_path + attack_type + '_' + tmp_method + '.png')
        plt.close()
    elif method == 'ae':
        plt.savefig(folder_path + 'binary' + '.png')
        plt.close()

def draw_roc_multi_test(file, ax, title):
    """
        For multi test
    """
    with open(file, 'rb') as f:
        result = pickle.load(f)
    score = np.concatenate([s for s in result['score'] if len(s) == 12])  # 4 agent samples
    label = np.concatenate([l for l in result['label'] if len(l) == 12])  # 4 agent samples

    import sys
    sys.path.append("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net")
    from utils.bh_procedure import BH
    sys.path.pop()

    dists = []
    dists.append(np.load("/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/match_costs_validation.npy"))
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
    ax.plot(fprs, tprs, label='ROC (area=%0.3f)' % auc(fprs, tprs))
    # ax.set_title(title)
    ax.tick_params(labelsize=size)
    ax.legend(fontsize=new_size)
    
def main():

    # draw_roc(attack_type, method='match', binary=True)
    # draw_roc(attack_type, method='ae', ae_type='residual', binary=True)

    draw_roc(attack_type, method='match')
    draw_roc(attack_type, method='ae', ae_type='residual')
    # draw_roc(attack_type, method='ae', ae_type='raw')

    # draw_roc_multi_test(MULTI_TEST, axes[1][1], 'multi_test')


if __name__ == '__main__':
    main()
 