from nuscenes.nuscenes import NuScenes
from icecream import ic
import numpy as np
from pyquaternion import Quaternion
from pathlib import Path
import pickle
import os 
from tqdm import tqdm


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def make_split_2(nusc, train_n=80, val_n=10, test_n=10):
    """ make split of trainset, valset, testset
        for v2x-sim 2.0
    """
    scene_num = len(nusc.scene)
    assert (train_n + val_n + test_n) == scene_num
    
    np.random.seed(234)
    perm = np.random.permutation(scene_num)

    train_split = perm[:train_n]
    val_split = perm[train_n:train_n+val_n]
    test_split = perm[train_n+val_n:train_n+val_n+test_n]

    ic(train_split)
    ic(val_split)
    ic(test_split)

    return train_split, val_split, test_split


def build_hash_map(nusc):

    instance_set = set()

    for sample in nusc.sample:
        anns = sample['anns']
        instance_token = [nusc.get("sample_annotation",anno_token)['instance_token'] for anno_token in anns]

        instance_set.update(instance_token)
    
    hash_map = dict()

    for idx, token in enumerate(instance_set):
        hash_map[token] = idx
    
    return hash_map



def fill_split_info(nusc, scene_ids):
    """ use the v2x-sim dataset in nuscenes format, and aggregate the point cloud file path,
        ego pose, gt boxes information. For one split(train/val/test).

        only support single sweep now.

    Args:
        data_path: str, root path for nusc dataset.  e.g. '/GPFS/rhome/yifanlu/workspace/dataset/v2xsim2-complete'
        nusc: nuscenes dataset object
        scene_ids: list

    Returns:
        split_infos: list  
    """
    split_infos = []

    for scene_id in tqdm(scene_ids):
        scene = nusc.scene[scene_id]
        sample_token = scene['first_sample_token']
        while(sample_token != ''):
            sample = nusc.get('sample', sample_token)  # dict
            agent_num = eval(max([i[-1] for i in sample['data'].keys() if i.startswith("LIDAR_TOP")])) 
            # LIDAR_TOP_id_0 is not vehicle. It's roadside unit.

            info = {
            'token': sample['token'],
            'timestamp': sample['timestamp'],
            'agent_num': agent_num, 
            }
            for i in range(1, agent_num + 1):
                lidar_sample_data =  nusc.get("sample_data", sample['data'][f'LIDAR_TOP_id_{i}']) # dict

                info[f'lidar_path_{i}'] = nusc.get_sample_data_path(lidar_sample_data['token'])

                # dict, include token, timestamp, rotation, translation. rotation is Quaternion
                ego_pose_record = nusc.get("ego_pose", lidar_sample_data['ego_pose_token'])
                q = Quaternion(ego_pose_record['rotation'])
                T_world_ego = q.transformation_matrix
                T_world_ego[:3,3] = ego_pose_record['translation']

                info[f'ego_pose_{i}'] = T_world_ego
                
                # dict, include token, timestamp, rotation, translation. rotation is Quaternion
                cs_record = nusc.get("calibrated_sensor", lidar_sample_data['calibrated_sensor_token'])
                translation_ego_lidar = cs_record['translation']
                rotation_ego_lidar = cs_record['rotation']
                q = Quaternion(rotation_ego_lidar)
                T_ego_lidar = q.transformation_matrix
                T_ego_lidar[:3,3] = translation_ego_lidar

                T_world_lidar = np.dot(T_world_ego, T_ego_lidar)
                info[f'lidar_pose_{i}'] = T_world_lidar


            # id 0 exists for sure, it will fetch all annotations in the sample
            # they are shared for all agents

            # boxes are in global coordinate
            boxes = nusc.get_boxes(sample['data']['LIDAR_TOP_id_1'])

            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            rots = np.array([b.orientation.elements for b in boxes]).reshape(-1, 4)
            names = np.array([b.name for b in boxes]) # like "vehicle.audi.tt"
            tokens = [b.token for b in boxes]
            object_ids = [hash_map[nusc.get("sample_annotation", anno_token)['instance_token']] for anno_token in tokens]
            tokens = np.array(tokens)
            object_ids = np.array(object_ids)

            gt_boxes = np.concatenate([locs, dims, rots], axis=1)  # [N, 10]

            vehicle_mask1 = [name.startswith("vehicle") for name in names]
            vehicle_mask2 = (dims[:,1] > 1.5).tolist() # filter small vehicle, width smaller than 1.5m.  Are they really vehicles?
            vehicle_mask = np.array([i and j for (i, j) in zip(vehicle_mask1,vehicle_mask2)], dtype=bool)

            info['gt_names'] = names[vehicle_mask]
            info['gt_boxes_token'] = tokens[vehicle_mask]
            info['gt_boxes_global'] = gt_boxes[vehicle_mask]
            info['gt_object_ids'] = object_ids[vehicle_mask]

            split_infos.append(info)
                
            sample_token = sample['next']

    return split_infos

def fill_infos(nusc, train_split, val_split, test_split):
    """ use the v2x-sim dataset in nuscenes format, and aggregate the point cloud file path,
    ego pose, gt boxes information.

    Args:
        data_path: str, root path for nusc dataset
        nusc: nuscenes dataset object
        *_split: list, contains index of scene
    """

    scene_num = len(nusc.scene)
    assert (len(train_split) + len(val_split) + len(test_split)) == scene_num

    train_infos = fill_split_info(nusc, train_split)
    val_infos = fill_split_info(nusc, val_split)
    test_infos = fill_split_info(nusc, test_split)

    return train_infos, val_infos, test_infos

def create_pkl(nusc, save_dir):
    """ create info.pkl for train/val/test set

    Args: 
        nusc: nuscenes dataset object
        save_dir: directory for saving pkl.

    Returns: None
    """
    train_split, val_split, test_split = make_split_2(nusc)
    train_infos, val_infos, test_infos = fill_infos(nusc, train_split, val_split, test_split)

    print(f'train sample: {len(train_infos)}, val sample: {len(val_infos)}, test sample: {len(test_infos)}')

    with open(os.path.join(save_dir, f'v2xsim_infos_train.pkl'), 'wb') as f:
        pickle.dump(train_infos, f)
    with open(os.path.join(save_dir, f'v2xsim_infos_val.pkl'), 'wb') as f:
        pickle.dump(val_infos, f)
    with open(os.path.join(save_dir, f'v2xsim_infos_test.pkl'), 'wb') as f:
        pickle.dump(test_infos, f)

def create_pkl_vis(nusc, output_path, split):
    """ create a single pkl for select scenes
    
    """
    split_info = fill_split_info(nusc, split)
    with open(os.path.join(output_path, f'v2xsim_infos_vis{split}.pkl'), 'wb') as f:
        pickle.dump(split_info, f)

if __name__=="__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='/dssg/home/acct-seecsh/seecsh/public_dataset/V2X-Sim/V2X-Sim-2.0', verbose=True) # specify your own path.
    output_path = "./v2xsim2_info" # a folder path. specify your own path.
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    global hash_map
    hash_map = build_hash_map(nusc)

    # create_pkl(nusc, output_path)  ## create pkl for train/val/test
    create_pkl_vis(nusc, output_path, split=[31])  ## create pkl for select scenes


# ic| train_split: array([13, 46, 89, 88, 99, 37, 49, 85,  6, 21, 65,  5, 17, 29, 28, 11, 97,
#                         79, 62, 70, 38, 35, 74, 86, 81, 94, 48, 96, 50, 58, 30, 43, 60,  1,
#                         44, 36, 82, 52, 42, 22, 12, 80, 56, 24, 66, 47,  4, 53, 27, 90, 25,
#                          2, 87, 16, 98, 61, 10,  0, 14, 69, 55, 91, 40,  9, 73, 41, 76,  7,
#                         18, 78, 20, 19, 51, 77, 75, 45, 15, 93, 23, 84])
# ic| val_split: array([26, 64, 59, 32, 71, 39, 83,  8, 92, 34])
# ic| test_split: array([67, 63, 54,  3, 33, 95, 57, 68, *31*, 72])

