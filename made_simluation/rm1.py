import numpy as np
import pickle
import os 
import torch
import ipdb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.match import HungarianMatcher, HungarianMatcherV2

if __name__ == "__main__":
    with open("experiments/rm1/no-attack/result.pkl", 'rb') as f:
        data = pickle.load(f)

    label = data['label']
    bboxes = data['bboxes']
    match_costs = []

    matcher = HungarianMatcherV2()
    # box_list = bboxes[0]
    for box_list in tqdm(bboxes):
        all_result = box_list[0]
        match_cost = [matcher(box_list[i], all_result) for i in range(1, len(box_list))]
        # print(match_cost)
        match_costs_tensor = torch.Tensor(match_cost)
        match_costs_tensor = match_costs_tensor[:, 0] if match_costs_tensor.ndim == 3 else match_costs_tensor
        match_costs_tensor = match_costs_tensor.reshape(-1)
        match_costs.append(match_costs_tensor.detach().cpu().numpy())
    
    # ipdb.set_trace()
    with open("experiments/rm1/no-attack/match_cost_v2.pkl", 'wb') as f:
        pickle.dump({
            "score": match_costs,
            "label": label,
        }, f)
    labels = np.concatenate(label)
    match_costs = np.concatenate(match_costs)
    
    match_costs[np.isnan(match_costs)] = 0
    plt.figure("RM1 match cost")
    plt.hist(match_costs[labels==0], bins=100, alpha=0.5, label="normal")
    plt.hist(match_costs[labels==1], bins=100, alpha=0.5, label="attack")
    plt.legend()
    plt.savefig("experiments/rm1/no-attack/match_cost_v2-2.png")
