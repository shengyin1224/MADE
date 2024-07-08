from typing import Dict, List
import numpy as np 
from .mean_ap import tpfp_default, average_precision
# mAP calculator 

__all__ = ['EvalWorker']


def get_cls_results(det_result, annotation, class_id):
    cls_det = det_result[class_id]
    cls_gt = annotation['bboxes'][annotation['labels'] == class_id, :]
    
    if "labels_ignore" in annotation:
        cls_gt_ignore = annotation['bboxes_ignore'][annotation['labels_ignore'] == class_id, :]
    else:
        cls_gt_ignore = np.empty((0, 8), dtype=np.float32)
    
    return cls_det, cls_gt, cls_gt_ignore

def compute_AP(tp: np.ndarray, fp: np.ndarray, score: np.ndarray, num_gt: int) -> float:
    
    sort_inds = np.argsort(-score) # decending order

    sorted_tp = tp[:, sort_inds]
    sorted_fp = fp[:, sort_inds]

    score_tp = np.cumsum(sorted_tp, axis=1)
    score_fp = np.cumsum(sorted_fp, axis=1)

    eps = np.finfo(np.float32).eps
    recalls = score_tp / np.maximum(num_gt, eps)
    precisions = score_tp / np.maximum((score_tp + score_fp), eps)

    recalls = recalls[0]
    precisions = precisions[0]
    mode = 'area'

    AP = average_precision(recalls, precisions, mode)
    return AP


class EvalWorker:
    """
        Evaluate Co-Peception Model
    """
    def __init__(self, iou_thrs: List[float], num_classes: int = 1, cache: bool = True):
        
        self.iou_thrs = iou_thrs
        self.num_classes = num_classes
        self.cache = cache
        
        if cache:
            self.det_result = dict()
            self.annoations = dict()
            self.tpfp = {
                thr: dict() for thr in self.iou_thrs
            }

    def evaluate(self, pred: List, gt: List, num_agents: int) -> Dict:
        """
            Per-sample evaluate, each sample include `num_agents` agents
        """
        if self.cache and num_agents not in self.det_result:
            self.det_result[num_agents] = [[] for _ in range(num_agents)]
            self.annoations[num_agents] = [[] for _ in range(num_agents)]
            for k, v in self.tpfp.items():
                v[num_agents] = [[{"tp": [], "fp": []} for _ in range(self.num_classes)] for _ in range(num_agents)]

        result_dict = dict()
        for i in range(num_agents):
            if self.cache:
                self.det_result[num_agents][i].append(pred[i])
                self.annoations[num_agents][i].append(gt[i])
            
            # 不同threshold
            for thr in self.iou_thrs:
                APs = []
                for c in range(self.num_classes):
                    cls_det, cls_gt, cls_gt_ignore = get_cls_results(pred[i], gt[i], c)

                    # 计算 tp fp
                    tp, fp = tpfp_default(cls_det, cls_gt, cls_gt_ignore, thr)
                    if self.cache:
                        self.tpfp[thr][num_agents][i][c]['tp'].append(tp)
                        self.tpfp[thr][num_agents][i][c]['fp'].append(fp)

                    # 计算当前frame, class c的AP
                    score = cls_det[:, -1]
                    AP = compute_AP(tp, fp, score, cls_gt.shape[0])
                    APs.append(AP)
                mAP = np.array(APs).mean()
                result_dict[f'agent_{i} mAP@{thr}'] = mAP
                # print
        return result_dict

    def summary(self) -> Dict:
        """
            Summary:
                1. all frames
                2. different agent nums
                3. each agent in different agent nums
        """
        summary_dict = dict()
        if not self.cache:
            raise ValueError("Cache is not enabled")
        
        for thr in self.iou_thrs:
            all_tp = [[] for _ in range(self.num_classes)]
            all_fp = [[] for _ in range(self.num_classes)]
            all_score = [[] for _ in range(self.num_classes)]
            all_gt_num = [0 for _ in range(self.num_classes)]
            for num_agents in self.det_result.keys():
                n_agent_tp = [[] for _ in range(self.num_classes)]
                n_agent_fp = [[] for _ in range(self.num_classes)]
                n_agent_score = [[] for _ in range(self.num_classes)]
                n_agent_gt_num = [0 for _ in range(self.num_classes)]
                for i in range(num_agents):
                    APs = []
                    for c in range(self.num_classes):
                        tp = np.hstack(self.tpfp[thr][num_agents][i][c]['tp']) 
                        fp = np.hstack(self.tpfp[thr][num_agents][i][c]['fp'])
                        score = np.hstack([d[c][:, -1] for d in self.det_result[num_agents][i]])
                        agent_gt_num = sum([sum(ann['labels'] == c) for ann in self.annoations[num_agents][i]])
                        n_agent_tp[c].append(tp)
                        n_agent_fp[c].append(fp)
                        n_agent_score[c].append(score)
                        n_agent_gt_num[c] += agent_gt_num

                        # calculate AP
                        AP = compute_AP(tp, fp, score, agent_gt_num)
                        APs.append(AP)
                    mAP = np.array(APs).mean()
                    summary_dict[f"mAP@{thr} agent {i} of {num_agents} agents"] = mAP

                APs = []
                for c in range(self.num_classes):
                    tp = np.hstack(n_agent_tp[c])
                    fp = np.hstack(n_agent_fp[c])
                    score = np.hstack(n_agent_score[c])

                    all_tp[c].append(tp)
                    all_fp[c].append(fp)
                    all_score[c].append(score)
                    all_gt_num[c] += n_agent_gt_num[c]

                    # calculate AP
                    AP = compute_AP(tp, fp, score, n_agent_gt_num[c])
                    APs.append(AP)
                mAP = np.array(APs).mean()
                summary_dict[f"mAP@{thr} {num_agents} agents"] = mAP
            
            APs = []
            for c in range(self.num_classes):
                tp = np.hstack(all_tp[c])
                fp = np.hstack(all_fp[c])
                score = np.hstack(all_score[c])

                AP = compute_AP(tp, fp, score, all_gt_num[c])
                APs.append(AP)
            mAP = np.array(APs).mean()
            summary_dict[f"mAP@{thr} all"] = mAP

        return summary_dict
