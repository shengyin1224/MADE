import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from utils.mean_average_precision import EvalWorker
from utils.utils import get_pred_box

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str, help="Path to the experiment directory")
    args = parser.parse_args()

    eval_worker = EvalWorker([0.5, ], 1, cache=False)
    with open(os.path.join(args.exp_dir, "result.pkl"), 'rb') as f:
        result = pickle.load(f)
    
    with open("gt_bboxes.pkl", 'rb') as f:
        gt_bboxes = pickle.load(f)['gt_bbox']
    
    bboxes = result['bboxes']
    all_APs = []
    agent4_APs = []
    for box_list, gt_box in tqdm(zip(bboxes, gt_bboxes), total=len(bboxes)):
        num_agent = len(box_list)
        APs = []
        for i in range(num_agent):
            eval_result = eval_worker.evaluate([get_pred_box(b) for b in box_list[i]], gt_box, num_agent)
            APs.append([eval_result[f"agent_{j} mAP@0.5"] for j in range(num_agent)])
        all_APs.append(APs)
        if num_agent == 4:
            agent4_APs.append(APs)
        # import ipdb;ipdb.set_trace()
    with open(os.path.join(args.exp_dir, "AP05.pkl"), 'wb') as f:
        pickle.dump({"all_APs": all_APs, "agent4_APs": agent4_APs}, f)