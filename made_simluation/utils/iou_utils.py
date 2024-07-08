import torch
import sys
# sys.path.append('/DB/data/yanghengzhao/adversarial/Rotated_IoU')
sys.path.append('thirdparty/Rotated_IoU')
from oriented_iou_loss import cal_diou, cal_giou, cal_iou

def sincos2angle(x: torch.Tensor):
    # torch.atan2(sin, cos)
    return torch.atan2(x[..., 0], x[..., 1]).unsqueeze(-1)