import os
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Feature saved dir")

    args = parser.parse_args()
    return args

def main(args):
    files = sorted(os.listdir(os.path.join(args.dir, 'feature')))
    save_path = os.path.join(args.dir, 'feature_vis')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file_name in tqdm(files):
        with open(os.path.join(args.dir, 'feature', file_name), 'rb') as f:
            feature = pickle.load(f)

        before_perturb = feature['before_perturb']
        after_perturb = feature['after_perturb']
        n_agents = len(before_perturb)

        fig = plt.figure(figsize=(5*n_agents,10))
        axes = fig.subplots(ncols=n_agents, nrows=2)
        # import ipdb;ipdb.set_trace()
        for i in range(n_agents):
            axes[0][i].set_xticks([])
            axes[0][i].set_yticks([])
            axes[0][i].imshow(before_perturb[i].sum(axis=0))
            axes[1][i].set_xticks([])
            axes[1][i].set_yticks([])
            axes[1][i].imshow(after_perturb[i].sum(axis=0))
        fig.savefig(os.path.join(save_path, file_name.replace('pkl', 'png')))
        plt.close(fig)
        # import ipdb;ipdb.set_trace()

        
if __name__ == "__main__":
    args = parse_args()
    main(args)