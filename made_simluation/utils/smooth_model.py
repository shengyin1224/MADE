from typing import Optional
from collections import Counter
import torch 
import scipy.stats as stats
import math
from utils.detection_util import center_to_corner_box2d_torch, deg2sincos
from utils.AttackCoDetModel import DiscoNet
from utils.denoiser import get_pretrained_denoiser

class SmoothMedianNMS(torch.nn.Module):
    def __init__(self, co_detector: DiscoNet, sigma: float, accumulator):
        super().__init__()
        self.model = co_detector
        self.sigma = sigma
        self.detection_acc = accumulator
        # TODO train a denoiser
        self.denoiser = torch.nn.Sequential()
        # self.denoiser = get_pretrained_denoiser()
        
        # self.q_u, self.q_l = estimated_qu_ql(eps=0.36, sample_count=200, sigma=sigma, conf_thres=0.99999)
        # print(self.q_l, self.q_u)
        self.q_u, self.q_l = 0, 0

    @torch.no_grad()
    def forward(self, n: int, 
                    bevs: torch.Tensor,
                    trans_matrices: torch.Tensor,
                    num_agent_tensor: torch.Tensor,
                    batch_size: int,
                    batch_anchors: torch.Tensor,
                    com_src: Optional[torch.Tensor] = None,
                    com_tgt: Optional[torch.Tensor] = None,
                    attack: Optional[torch.Tensor] = None,
                    attack_src: Optional[torch.Tensor] = None,
                    attack_tgt: Optional[torch.Tensor] = None,
                    ):
        if com_src is None or com_tgt is None:
            assert num_agent_tensor is not None, f"`num_agent_tensor`, `com_src` and `com_tgt` cannot be None at the same time"
            com_src, com_tgt = self.model.get_default_com_pair(num_agent_tensor)

        # 拆开 model 的 forward
        feat_maps = list(self.model.encode(bevs))
        o_feat = feat_maps[self.model.layer].clone()

        # TODO support batch_size > 1
        for _ in range( n // batch_size):
            feat_maps[self.model.layer] = self.denoiser(o_feat + torch.randn_like(o_feat) * self.sigma)
            _, detections = self.model.after_encode(feat_maps, trans_matrices, com_src, com_tgt, 
                                                    attack, attack_src, attack_tgt, batch_size, 
                                                    batch_anchors, nms=True)
            # import ipdb;ipdb.set_trace()
            self.detection_acc.track(detections)

        self.detection_acc.tensorize()

        detections = self.detection_acc.median()
        detections_l = self.detection_acc.k(self.q_l)
        detections_u = self.detection_acc.k(self.q_u)
        self.detection_acc.clear()
        # import ipdb;ipdb.set_trace()
        return detections, detections_u, detections_l
        

class DetectionsAcc:
    OBJECT_SORT=0
    CENTER_SORT=1
    SINGLE_BIN=0
    LABEL_BIN=1
    LOCATION_BIN=2
    LOCATION_LABEL_BIN=3

    def __init__(self, bin=LOCATION_LABEL_BIN, sort=OBJECT_SORT, loc_bin_count=None):
        self.detections_list = [[] for _ in range(5)]  # 5 agents
        self.max_num_detections = [0 for _ in range(5)]
        #count the number of classes in each class bin
        self.bin_counts = [{} for _ in range(5)]
        self.detections_tensor = [None for _ in range(5)]
        self.id_index_map = [{} for _ in range(5)]

        self.sort = sort
        self.bin = bin
        self.loc_bin_count = loc_bin_count

    def track(self, detections):
        #dim of detections (# of simulations, tensor((#of detections, 7)))

        # 把 detections 表示成 (x, y, w, l, theta, score, class) 的形式

        for i, det_result in enumerate(detections):
            # FIXME 可能需要处理检测结果为空的情况
            det_vector = det_result_to_vector(det_result)
            self.detections_list[i].append(det_vector)

            if det_vector.shape[0] > 0:
                # temp_count = {}
                if self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
                    x = (det_vector[:, 0] + 32) / 0.25
                    y = (det_vector[:, 1] + 32) / 0.25

                    # assert (x <= 256).all() and (x >= 0).all() and (y <= 256).all() and (y >= 0).all()
                    x = x.clamp(0, 256)
                    y = y.clamp(0, 256)

                    xids = (x / 256 * self.loc_bin_count).floor()
                    yids = (y / 256 * self.loc_bin_count).floor()
                    labels = det_vector[:, -1]
                    ids = (xids + yids * 10 + labels * 100).tolist()

                    temp_count = Counter(ids)
                    for id, count in temp_count.items():
                        if id not in self.bin_counts[i]:
                            self.bin_counts[i][id] = count
                        elif self.bin_counts[i][id] < count:
                            self.bin_counts[i][id] = count
                else:
                    raise NotImplementedError(self.bin)

    def tensorize(self):
        if self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
            self.detection_len = [0 for _ in range(5)]
            for i in range(5):
                for id, count in self.bin_counts[i].items():
                    self.id_index_map[i][id] = self.detection_len[i]
                    self.detection_len[i] += count
        else:
            raise NotImplementedError(self.bin)

        self.detections_tensor = [torch.ones(
                (len(self.detections_list[i]), self.detection_len[i], 7)
            )*float('inf') 
            for i in range(5)]

        for agent in range(5):
            for i, detection in enumerate(self.detections_list[agent]):
                if len(detection) > 0:
                    if self.sort == DetectionsAcc.OBJECT_SORT:
                        detection_count = detection.size(0)
                    elif self.sort == DetectionsAcc.CENTER_SORT:
                        detection_count = detection.size(0)
                        midy = detection[:, 1]
                        _, sort_idx = midy.sort(dim=0)
                        detection = detection[sort_idx]
                        midx = detection[:, 0]
                        _, sort_idx = midx.sort(dim=0)
                        detection = detection[sort_idx]  # 这不就是按x排序了吗？

                    if self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
                        x = (detection[:, 0] + 32) / 0.25
                        y = (detection[:, 1] + 32) / 0.25

                        x = x.clamp(0, 256)
                        y = y.clamp(0, 256)

                        xids = (x / 256 * self.loc_bin_count).floor()
                        yids = (y / 256 * self.loc_bin_count).floor()
                        labels = detection[:, -1]
                        ids = xids + yids * 10 + labels * 100
                        unique_ids = ids.unique()

                    for id in unique_ids:
                        filtered_detection = detection[ids == id]
                        filtered_len = filtered_detection.size(0)
                        idx_st = self.id_index_map[agent][id.cpu().item()]
                        self.detections_tensor[agent][i, idx_st:idx_st+filtered_len]= filtered_detection

            self.detections_tensor[agent], _ = self.detections_tensor[agent].sort(dim=0)

    def median(self):
        result = [vector_to_det_result(self.detections_tensor[i][len(self.detections_list[i]) // 2] )
                    for i in range(5) if len(self.detections_list[i]) > 0]
        return result

    def upper(self, alpha=.05):
        result = [vector_to_det_result(self.detections_tensor[i][int(len(self.detections_list[i])*(alpha))])
                    for i in range(5) if len(self.detections_list[i]) > 0]
        return result

    def lower(self, alpha=.05):
        result = [vector_to_det_result(self.detections_tensor[i][int(len(self.detections_list[i])*(1-alpha))])
                    for i in range(5) if len(self.detections_list[i]) > 0]
        return result

    def k(self, q):
        result = [vector_to_det_result(self.detections_tensor[i][q] )
                    for i in range(5) if len(self.detections_list[i]) > 0]
        return result

    def clear(self):
        self.detections_list = [[] for _ in range(5)]
        self.max_num_detections = [0 for _ in range(5)]
        self.detections_tensor = [None for _ in range(5)]
        # FIXME bin_count 不清理吗  清掉
        self.bin_counts = [{} for _ in range(5)]

def estimated_qu_ql(eps, sample_count, sigma, conf_thres = .99999):
    theo_perc_u = stats.norm.cdf(eps/sigma)
    theo_perc_l = stats.norm.cdf(-eps / sigma)

    q_u_u = sample_count + 1
    q_u_l = math.ceil(theo_perc_u*sample_count)
    q_l_u = math.floor(theo_perc_l*sample_count)
    q_l_l = 0
    q_u_final = q_u_u
    for q_u in range(q_u_l, q_u_u):
        conf = stats.binom.cdf(q_u-1, sample_count, theo_perc_u)
        if conf > conf_thres:
            q_u_final = q_u
            break

    q_l_final = q_l_l
    for q_l in range(q_l_u, q_l_l, -1):
        conf = 1-stats.binom.cdf(q_l-1, sample_count, theo_perc_l)
        if conf > conf_thres:
            q_l_final = q_l
            break

    return q_u_final, q_l_final



def det_result_to_vector(det_result):
    vectors = []
    for class_id, result_d in enumerate(det_result):
        box = result_d['rot_box']
        score = torch.from_numpy(result_d['score']).to(box.device)
        c = torch.ones_like(score) * class_id
        vector = torch.cat([box, score[:, None], c[:, None]], dim=1)
        vectors.append(vector)
    return torch.cat(vectors, dim=0)

def vector_to_det_result(det_vector, num_classes=1):
    det_result = []
    for i in range(num_classes):
        class_idx = det_vector[:, -1] == i
        pred = center_to_corner_box2d_torch(det_vector[class_idx, :2], det_vector[class_idx, 2:4], deg2sincos(det_vector[class_idx, 4]))
        det_result.append({'rot_box': det_vector[class_idx, :-2],
                           'score': det_vector[class_idx, -2].detach().cpu().numpy(),
                           'pred': pred.unsqueeze(1).detach().cpu().numpy()})
        # CAUTION: selected_idx is dropped.
    return det_result

if __name__ == "__main__":
    print("Debug module")
    
    input_dict = torch.load("/DB/data/yanghengzhao/adversarial/DiscoNet/debug_smooth.pth")
    smooth_model = torch.load("/DB/data/yanghengzhao/adversarial/DiscoNet/smooth_model.pth")

    _, det_result = smooth_model.model(**input_dict, nms=True)

    a0_det_result = det_result[0]
    vec = det_result_to_vector(a0_det_result)
    rec_det_result = vector_to_det_result(vec)
    # smooth_model(n=1, **input_dict)

    # import ipdb;ipdb.set_trace()