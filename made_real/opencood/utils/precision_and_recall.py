import numpy as np

def compute_precision_and_recall(pred_and_label_list):

    # if method == 'match' or method == 'ae':
    #     pred = score > threshold
    # elif method == 'multi_test_or':
    #     #此时score和threshold均为列表
    #     pred1 = (score[0] - threshold[0]) > 0
    #     pred2 = (score[1] - threshold[1]) > 0
    #     pred = pred1 | pred2 + 0
    # else:
    #     pred = np.array(pred)

    pred = np.array([x[0] for x in pred_and_label_list])
    label = np.array([x[1] for x in pred_and_label_list])
    
    tp = ((label == 1) & (pred == 1)).sum()
    tn = ((label == 0) & (pred == 0)).sum()
    fp = ((label == 0) & (pred == 1)).sum()
    fn = ((label == 1) & (pred == 0)).sum()

    print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    precision = tp / (tp + fp + 1e-7) 
    recall = tp / (tp + fn + 1e-7) 
    fpr = fp / (fp + tn + 1e-7)

    return precision, recall, fpr, tp, tn, fp, fn