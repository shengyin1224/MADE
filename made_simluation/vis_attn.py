import os
import glob
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from data.config import Config

config = Config('test', binary=True, only_det=True)
config.flag = 'disco'
voxel_size = config.voxel_size
area_extents = config.area_extents

def rescale_bboxes(voxel_size, area_extents, bboxes):
    bboxes = bboxes.reshape(-1, 4, 2)  
    bboxes = bboxes.copy()  # copy
    bboxes[..., 0] = (bboxes[..., 0] - area_extents[0][0]) / voxel_size[0] / 8
    bboxes[..., 1] = (bboxes[..., 1] - area_extents[1][0]) / voxel_size[1] / 8
    return bboxes

def basic_box_draw(ax, bboxes, color, cfd):
    # for bboxes, color in bboxes_color:
    for bbox, c in zip(bboxes, cfd):
        polygon = patches.Polygon(bbox, fill=False, edgecolor=color, linewidth=2.5)
        ax.add_patch(polygon)
        ax.text(
            bbox[0, 0], bbox[0, 1], f"{c:.2f}",
            color='black', fontsize=10, fontweight='bold'
        )

def plot_attn_map():
    cmap = mpl.cm.get_cmap('gist_heat')
    scene_name = "84_14"
    attn_maps = np.load(f"vis_vis/no_attack/raw/{scene_name}.npy")
    n_agents = int(np.sqrt(attn_maps.shape[0]))
    self_attn_maps = attn_maps[::n_agents]

    agent_id = 0

    img1 = cmap(self_attn_maps[agent_id, 0])
    img1 = Image.fromarray(np.uint8(img1 * 255))
    img1.save(f"vis_vis/no_attack/png/{scene_name}_{agent_id}.png")

    plt.imshow(self_attn_maps[agent_id, 0], cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.savefig(f"vis_vis/no_attack/png/{scene_name}_{agent_id}.jpg")

def plot_bbox(scene_name, folder):
    cmap = mpl.cm.get_cmap('gist_heat')
    # scene_name = "84_14"
    attn_maps = np.load(f"vis_vis/{folder}/raw/{scene_name}.npy")
    n_agents = int(np.sqrt(attn_maps.shape[0]))
    self_attn_maps = attn_maps[::n_agents]

    with open(f"vis_vis/{folder}/raw/{scene_name}.pkl", 'rb') as f:
        data = pickle.load(f)

    fig = plt.figure(figsize=(7*n_agents, 8))
    axes = fig.subplots(ncols=n_agents)

    for agent_id in range(n_agents):
        im = axes[agent_id].imshow(self_attn_maps[agent_id, 0], cmap=cmap)
        axes[agent_id].set_xticks([])
        axes[agent_id].set_yticks([])
    
    # import ipdb;ipdb.set_trace()
        pred_bboxes = data['pred_bbox'][agent_id][0]
        bboxes, cfd = pred_bboxes[:, :-1], pred_bboxes[:, -1]
        bboxes = rescale_bboxes(voxel_size, area_extents, bboxes)
        basic_box_draw(axes[agent_id], bboxes, '#00FA65', cfd)
    
    fig.colorbar(im, ax=axes)
    # fig.tight_layout()
    fig.savefig(f"vis_vis/{folder}/png/{scene_name}.jpg")
    plt.close(fig)
    plt.close(fig)


if __name__ == "__main__":
    # folder = "no_attack"
    folder = "attack"
    file_names = glob.glob(f"vis_vis/{folder}/raw/*.pkl")
    scene_names = [f.split('/')[-1][:-4] for f in file_names]
    for s in tqdm(scene_names):
        plot_bbox(s, folder)