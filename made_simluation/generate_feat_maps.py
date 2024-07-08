import matplotlib
matplotlib.use('Agg')
import random
import torch
import torch.optim as optim
from copy import deepcopy
from collections import defaultdict
import argparse
from utils.CoDetModule import *
# from utils.CoDetModel import *
from utils.AttackCoDetModel import *
from utils.loss import *
from data.Dataset import V2XSIMDataset
from data.config import Config, ConfigGlobal
from utils.mean_ap import eval_map
from utils.utils import get_pred_box, get_gt_box
from tqdm import tqdm
from terminaltables import AsciiTable
from utils.mean_average_precision import EvalWorker


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def main(args):
    config = Config('train', binary=True, only_det=True)
    config_global = ConfigGlobal('train', binary=True, only_det=True)

    need_log = args.log
    num_workers = args.nworker

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    if args.bound == 'upperbound':
        flag = 'upperbound'
    else:
        if args.com == 'when2com':
            flag = 'when2com'
            if args.inference == 'argmax_test':
                flag = 'who2com'
            if args.warp_flag:
                flag = flag + '_warp'
        elif args.com == 'v2v':
            flag = 'v2v'
        elif args.com == 'disco':
            flag = 'disco'
        elif args.com == 'sum':
            flag = 'sum'
        elif args.com == 'mean':
            flag = 'mean'
        elif args.com == 'max':
            flag = 'max'
        elif args.com == 'cat':
            flag = 'cat'
        elif args.com == 'agent':
            flag = 'agent'
        else:
            flag = 'lowerbound'
            if args.box_com:
                flag += '_box_com'

    print('flag', flag)
    config.flag =  flag
    config.split = 'test'
    valset = V2XSIMDataset(dataset_roots=[f'{args.data}/{args.split}/agent{i}' for i in range(5)], config=config, config_global=config_global, split='val', val=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)
    print("Validation dataset size:", len(valset))
    # trainset = V2XSIMDataset(dataset_roots=[f'{args.data}/agent{i}' for i in range(5)], config=config, config_global=config_global, split='train')
    # trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=4, num_workers=num_workers)

    logger_root = args.logpath if args.logpath != '' else 'logs'

    if flag == 'upperbound' or flag.startswith('lowerbound'):
        model = FaFNet(config)
    elif flag.startswith('when2com') or flag.startswith('who2com'):
        # model = PixelwiseWeightedFusionSoftmax(config, layer=args.layer)
        model = When2com(config, layer=args.layer, warp_flag=args.warp_flag)
    elif args.com == 'disco':
        model = DiscoNet(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'sum':
        model = SumFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'mean':
        model = MeanFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'max':
        model = MaxFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'cat':
        model = CatFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'agent':
        model = AgentwiseWeightedFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    else:
        model = V2VNet(config, gnn_iter_times=args.gnn_iter_times, layer=args.layer, layer_channel=256)

    print("Model created")
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {'cls': SoftmaxFocalClassificationLoss(), 'loc': WeightedSmoothL1LocalizationLoss()}

    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag, 
                        attack=args.attack, com=args.attack_com, attack_mode=args.attack_mode, 
                        eps=args.eps, alpha=args.alpha, proj=not args.attack_no_proj,
                        attack_target=args.att_target, vis_path=args.att_vis, step=args.step,
                        smooth=True, smooth_sigma=0.36)

    model_save_path = args.resume[:args.resume.rfind('/')]

    log_file_name = os.path.join(model_save_path, 'log.txt')
    saver = open(log_file_name, "a")
    saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
    saver.flush()

    # Logging the details for this experiment
    saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
    saver.write(args.__repr__() + "\n\n")
    saver.flush()

    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    fafmodule.model.load_state_dict(checkpoint['model_state_dict'])
    fafmodule.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    fafmodule.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    #  ===== eval =====
    fafmodule.model.eval()
    save_fig_path = [check_folder(os.path.join(model_save_path, f'vis{i}')) for i in range(5)]
    tracking_path = [check_folder(os.path.join(model_save_path, f'tracking{i}')) for i in range(5)]

    det_results = defaultdict(list)
    annotations = defaultdict(list)

    eval_worker = EvalWorker(iou_thrs=[0.5, 0.7], num_classes=1)

    # create save dir
    if not os.path.exists(os.path.join("/GPFS/data/yanghengzhao-1/adversarial/disco_features", args.split, args.save_path)):
        os.makedirs(os.path.join("/GPFS/data/yanghengzhao-1/adversarial/disco_features", args.split, args.save_path))

    tracking_file = [set()] * 5
    for cnt, sample in tqdm(enumerate(valloader), total=len(valloader)):
        t = time.time()
        padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, gt_max_iou, filenames, \
                target_agent_id_list, num_agent_list, trans_matrices_list = zip(*sample)

        filename0 = filenames[0]
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_agent = torch.stack(tuple(num_agent_list), 1)
        if flag == 'upperbound':
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        
        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        data = {}
        data['bev_seq'] = padded_voxel_points.to(device)
        data['labels'] = label_one_hot.to(device)
        data['reg_targets'] = reg_target.to(device)
        data['anchors'] = anchors_map.to(device)
        data['vis_maps'] = vis_maps.to(device)
        data['reg_loss_mask'] = reg_loss_mask.to(device).type(dtype=torch.bool)

        data['target_agent_ids'] = target_agent_ids.to(device)
        data['num_agent'] = num_agent.to(device)
        data['trans_matrices'] = trans_matrices.to(device)

        # features = model.module.encode(data['bev_seq'])

        filename = filename0[0][0]
        scene, frame = filename.split('/')[-2].split('_')

        result = fafmodule.generate_feature_map(data, 1, merge_feature=False)
        if isinstance(result, torch.Tensor):
            features = result
            num_sensor = num_agent_list[0][0].item()
            for k in range(num_sensor):
                agent_feat = features[k]
                # torch.save(agent_feat, 
                # os.path.join("../disco_features", args.split, "normal" f"{scene}_{frame}_{k}.pth"))
        else:
            features, src, tgt = result
            for i in range(len(features)):
                agent_feat = features[i]
                if fafmodule.attack:
                    save_file_name = os.path.join("/GPFS/data/yanghengzhao-1/adversarial/disco_features", args.split, args.save_path, f"{scene}_{frame}_{src[i, 1].item()}_{tgt[i, 1].item()}.pth")
                else:
                    save_file_name = os.path.join("/GPFS/data/yanghengzhao/adversarial/disco_features", args.split, args.save_path, f"{scene}_{frame}_{src[i, 1].item()}_{tgt[i, 1].item()}.pth")
                # import ipdb;ipdb.set_trace()
                torch.save(agent_feat, save_file_name)

        # import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default="../v2x-sim-1.0/", type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('-s', '--split', default='test', type=str, help='The split of the data to use (train, test)')
    parser.add_argument('--batch', default=4, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--nworker', default=1, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./log', help='The path to the output log file')
    parser.add_argument('--resume', default='checkpoints/DiscoNet.pth', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--resume_teacher', default='', type=str, help='The path to the saved teacher model that is loaded to resume training')
    parser.add_argument('--layer', default=3, type=int, help='Communicate which layer in the single layer com mode')
    parser.add_argument('--warp_flag', action='store_true', help='Whether to use pose info for Ｗhen2com')
    parser.add_argument('--kd_flag', default=0, type=int, help='Whether to enable distillation (only DiscNet is 1 )')
    parser.add_argument('--kd_weight', default=100000, type=int, help='KD loss weight')
    parser.add_argument('--gnn_iter_times', default=3, type=int, help='Number of message passing for V2VNet')
    parser.add_argument('--visualization', default=True, help='Visualize validation result')
    parser.add_argument('--com', default='disco', type=str, help='disco/when2com/v2v/sum/mean/max/cat/agent')
    parser.add_argument('--bound', type=str, default='lowerbound', help='The input setting: lowerbound -> single-view or upperbound -> multi-view')
    parser.add_argument('--inference', type=str)
    parser.add_argument('--tracking',action='store_true')
    parser.add_argument('--box_com', action='store_true')

    parser.add_argument('--attack', type=str, default=False, help="Attack config file")
    parser.add_argument('--step', type=int, default=15, help="attack step")
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--attack_no_proj', action='store_true', help='Flag to not use projection in attack')
    parser.add_argument('--attack_mode', type=str, default='self', help='Attack mode: [self, others]')
    parser.add_argument('--attack_com', type=bool, default=True, help="whether communicate")
    parser.add_argument('--no_com', action='store_false', dest="attack_com")
    parser.add_argument('--att_target', type=str, default='pred', help="use gt or predicted result as target")
    parser.add_argument('--att_vis', type=str, default=None, help='save path for visualize attacked feature maps')

    parser.add_argument("--save_path", type=str)

    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    print(args)
    main(args)