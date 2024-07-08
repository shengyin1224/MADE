import os
import glob
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.patches as patches

def visualize_no_attack(file_name):
    with open(os.path.join("vis_vis2/no_attack/raw/", file_name), 'rb') as f:
        data = pickle.load(f)

    feature = data['feature']
    fused = data['fused']
    com_src = data['com_src']
    com_tgt = data['com_tgt']
    reduce_tgt = data['reduce_tgt']

    num_agents = fused.shape[0]

    if num_agents != 4:
        return
    scene_name = file_name[:-4]
    attacker_feature = feature[1::num_agents]
    ego_feature = feature[0::num_agents]

    cmap = 'viridis'
    # cmap = mpl.cm.get_cmap('gist_heat')
    if not os.path.exists("vis_vis2/no_attack/png_0315/"):
        os.makedirs("vis_vis2/no_attack/png_0315/")

    # import pdb; pdb.set_trace()

    for i in range(num_agents):
        plt.figure()
        plt.imshow(attacker_feature[i].mean(axis=0), cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("feature map")
        plt.savefig(f"vis_vis2/no_attack/png_0315/{scene_name}_{i}_attacker.jpg")
        plt.close()

        plt.figure()
        plt.imshow(fused[i].mean(axis=0), cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("fused feature map (w/o attack)")
        plt.savefig(f"vis_vis2/no_attack/png_0315/{scene_name}_{i}_fused.jpg")
        plt.close()

        plt.figure()
        plt.imshow((fused[i] - ego_feature[i]).mean(axis=0), cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("feature map residual (w/o attack)")
        plt.savefig(f"vis_vis2/no_attack/png_0315/{scene_name}_{i}_res.jpg")
        plt.close()
    # import ipdb;ipdb.set_trace()


def visualize_attack(file_name):
    with open(os.path.join("vis_vis2/attack/raw_0315/", file_name), 'rb') as f:
        data = pickle.load(f)

    feature = data['feature']
    fused = data['fused']
    com_src = data['com_src']
    com_tgt = data['com_tgt']
    reduce_tgt = data['reduce_tgt']

    num_agents = fused.shape[0]
    scene_name = file_name[:-4]
    attacker_feature = feature[1::num_agents]
    ego_feature = feature[0::num_agents]

    # cmap = mpl.cm.get_cmap('gist_heat')
    cmap = 'viridis'
    if not os.path.exists("vis_vis2/attack/png_0315/"):
        os.makedirs("vis_vis2/attack/png_0315/")

    for i in range(num_agents):
        plt.figure()
        plt.imshow(attacker_feature[i].mean(axis=0), cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("attacker feature map")
        plt.savefig(f"vis_vis2/attack/png_0315/{scene_name}_{i}_attacker.jpg")
        plt.close()

        plt.figure()
        plt.imshow(fused[i].mean(axis=0), cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("fused feature map (w attack)")
        plt.savefig(f"vis_vis2/attack/png_0315/{scene_name}_{i}_fused.jpg")
        plt.close()

        plt.figure()
        plt.imshow((fused[i] - ego_feature[i]).mean(axis=0), cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("feature map residual (w attack)")
        plt.savefig(f"vis_vis2/attack/png_0315/{scene_name}_{i}_res.jpg")
        plt.close()


if __name__ == "__main__":
    file_list = os.listdir("vis_vis2/no_attack/raw/")
    for f in file_list:
        visualize_no_attack(f)
        # visualize_attack(f)