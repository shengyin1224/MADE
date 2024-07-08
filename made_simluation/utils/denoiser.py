import torch

import sys 
sys.path.append("thirdparty/denoise/code")
from my_dataset import get_feature_map_dataset, get_feature_map_denoiser_arch

def get_pretrained_denoiser(path="/DB/data/yanghengzhao/adversarial/denoised-smoothing/disconet_denoiser/feature/mse_obj/dncnn/noise_0.25/checkpoint.pth.tar"):
    model = get_feature_map_denoiser_arch()
    model.load_state_dict(torch.load(path, map_location="cpu")['state_dict'])
    model.eval()
    return model

