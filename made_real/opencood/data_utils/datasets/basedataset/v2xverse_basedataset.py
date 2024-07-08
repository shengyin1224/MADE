
import os
from collections import OrderedDict
import cv2
import h5py
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import re
import math

import logging
_logger = logging.getLogger(__name__)

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor


class V2XVERSEBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if self.train:
            root_dir = params['root_dir']
            towns = [1,2,3,4,6]
        elif not visualize:
            root_dir = params['validate_dir']
            towns = [7,10] # [6,7,8,9,10]
        else:
            root_dir = params['test_dir']
            towns = [5]
        self.root_dir = root_dir 
        
        # towns = [1,2]
        self.clock = 0

        print("Dataset dir:", root_dir)

        # self.det_range = params['loss']['args']['target_assigner_config']['cav_lidar_range']
        self.fusion_mode = 'inter'

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_folders = scenario_folders

        #################################
        ## v2xverse data load
        #################################

        self.rsu_change_frame = 25
        self.route_frames = []
        dataset_indexs = self._load_text('dataset_index.txt').split('\n')

        weathers = [0,1,2,3,4,5,6,7,8,9,10]
        

        pattern = re.compile('weather-(\d+).*town(\d\d)')
        for line in dataset_indexs:
            if len(line.split()) != 3:
                continue
            path, frames, egos = line.split()
            route_path = os.path.join(self.root_dir, path)
            frames = int(frames)
            res = pattern.findall(path)
            if len(res) != 1:
                continue
            weather = int(res[0][0])
            town = int(res[0][1])            
            if weather not in weathers or town not in towns:
                continue

            files = os.listdir(route_path)
            ego_files = [file for file in files if file.startswith('ego')]
            rsu_files = [file for file in files if file.startswith('rsu')]

            # recompute rsu change frames
            file_len_list = []
            if len(rsu_files) > 0:
                for rsu_file in ['rsu_1000', 'rsu_1001']:
                    if rsu_file in rsu_files:
                        rsu_frame_len = len(os.listdir(os.path.join(route_path,rsu_file,'measurements')))
                        file_len_list.append(rsu_frame_len)
            self.rsu_change_frame = max(file_len_list) + 1
            print(self.rsu_change_frame)

            for j, file in enumerate(ego_files):
                ego_path = os.path.join(route_path, file)
                others_list = ego_files[:j]+ego_files[j+1:]
                others_path_list = []
                for others in others_list:
                    others_path_list.append(os.path.join(route_path, others))

                for i in range(frames):
                    scene_dict = {}
                    scene_dict['ego'] = ego_path
                    scene_dict['other_egos'] = others_path_list
                    scene_dict['num_car'] = len(ego_files)
                    scene_dict['rsu'] = []
                    # order of rsu
                    if i%self.rsu_change_frame != 0  and len(rsu_files)>0:
                        order = int(i/self.rsu_change_frame)+1 #  int(i/10)+1 
                        rsu_path = 'rsu_{}00{}'.format(order, ego_path[-1])
                        if True: # os.path.exists(os.path.join(route_path, rsu_path,'measurements','{}.json'.format(str(i).zfill(4)))):
                            scene_dict['rsu'].append(os.path.join(route_path, rsu_path))

                    self.route_frames.append((scene_dict, i))
        print("Sub route dir nums: %d" % len(self.route_frames))

    def _load_text(self, path):
        text = open(os.path.join(self.root_dir,path), 'r').read()
        return text

    def _load_image(self, path):
        trans_totensor = torchvision.transforms.ToTensor()
        trans_toPIL = torchvision.transforms.ToPILImage()
        try:
            img = Image.open(os.path.join(self.root_dir,path))
            img_tensor = trans_totensor(img)
            img_PIL = trans_toPIL(img_tensor)
        except Exception as e:
            _logger.info(path)
            n = path[-8:-4]
            new_path = path[:-8] + "%04d.jpg" % (int(n) - 1)
            img = Image.open(os.path.join(self.root_dir,new_path))
            img_tensor = trans_totensor(img)
            img_PIL = trans_toPIL(img_tensor)
        return img_PIL

    def _load_json(self, path):
        try:
            json_value = json.load(open(os.path.join(self.root_dir,path)))
        except Exception as e:
            _logger.info(path)
            n = path[-9:-5]
            new_path = path[:-9] + "%04d.json" % (int(n) - 1)
            json_value = json.load(open(os.path.join(self.root_dir,new_path)))
        return json_value

    def _load_npy(self, path):
        try:
            array = np.load(os.path.join(self.root_dir,path), allow_pickle=True)
        except Exception as e:
            _logger.info(path)
            n = path[-8:-4]
            new_path = path[:-8] + "%04d.npy" % (int(n) - 1)
            array = np.load(os.path.join(self.root_dir,new_path), allow_pickle=True)
        return array

    def get_one_record(self, route_dir, frame_id, agent='ego', visible_actors=None, tpe='all', extra_source=None):
        '''
        Parameters
        ----------
        scene_dict: str, index given by dataloader.
        frame_id: int, frame id.

        Returns
        -------
        data:  
            structure: dict{
                ####################
                # input to the model
                ####################
                'agent': 'ego' or 'other_ego', # whether it is the ego car
                'rgb_[direction]': torch.Tenser, # direction in [left, right, center], shape (3, 128, 128)
                'rgb': torch.Tensor, front rgb image , # shape (3, 224, 224) 
                'measurements': torch.Tensor, size [7]: the first 6 dims is the onehot vector of command, and the last dim is car speed
                'command': int, 0-5, discrete command signal 0:left, 1:right, 2:straight, 
                                                    # 3: lane follow, 4:lane change left, 5: lane change right
                'pose': np.array, shape(3,), lidar pose[gps_x, gps_y, theta]
                'detmap_pose': pose for density map
                'target_point': torch.Tensor, size[2], (x,y) coordinate in the left hand coordinate system,
                                                                 where X-axis towards right side of the car
                'lidar': np.ndarray, # shape (3, 224, 224), 2D projection of lidar, range x:[-28m, 28m], y:[-28m,28m]
                                        in the right hand coordinate system with X-axis towards left of car
                ####################
                # target of model
                ####################
                'img_traffic': not yet used in model
                'command_waypoints': torch.Tensor, size[10,2], 10 (x,y) coordinates in the same coordinate system with target point
                'is_junction': int, 0 or 1, 1 means the car is at junction
                'traffic_light_state': int, 0 or 1
                'det_data': np.array, (400,7), flattened density map, 7 feature dims corresponds to 
                                                [prob_obj, box bias_X, box bias_Y, box_orientation, l, w, speed]
                'img_traj': not yet used in model
                'stop_sign': int, 0 or 1, exist of stop sign
        },
        '''

        output_record = OrderedDict()

        if agent == 'ego':
            output_record['ego'] = True
        else:
            output_record['ego'] = False

        BEV = None

        if route_dir is not None:
            measurements = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % frame_id))
            actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % frame_id))
        elif extra_source is not None:
            actors_data = extra_source['actors_data']
            measurements = extra_source['measurements']

        output_record['params'] = {}
        
        cam_list = ['front','right','left','rear']
        cam_angle_list = [0, 60, -60, 180]
        for cam_id in range(4):
            output_record['params']['camera{}'.format(cam_id)] = {}
            output_record['params']['camera{}'.format(cam_id)]['cords'] = [measurements['x'], measurements['y'], 1.0,\
	 						                                                0,measurements['theta']/np.pi*180+cam_angle_list[cam_id],0]
            output_record['params']['camera{}'.format(cam_id)]['extrinsic'] = measurements['camera_{}_extrinsics'.format(cam_list[cam_id])]
            output_record['params']['camera{}'.format(cam_id)]['intrinsic'] = measurements['camera_{}_intrinsics'.format(cam_list[cam_id])]

        if 'speed' in measurements:
            output_record['params']['ego_speed'] = measurements['speed']*3.6
        else:
            output_record['params']['ego_speed'] = 0

        output_record['params']['lidar_pose'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                        0,measurements['theta']/np.pi*180-90,0]
        output_record['params']['lidar_pose_clean'] = output_record['params']['lidar_pose']
        output_record['params']['plan_trajectory'] = []
        output_record['params']['true_ego_pos'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                         0,measurements['theta']/np.pi*180,0]
        output_record['params']['predicted_ego_pos'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                        0,measurements['theta']/np.pi*180,0]
        
        if tpe == 'all':
            if route_dir is not None:
                lidar = self._load_npy(os.path.join(route_dir, "lidar", "%04d.npy" % frame_id))
                output_record['rgb_front'] = self._load_image(os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id))
                output_record['rgb_left'] = self._load_image(os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id))
                output_record['rgb_right'] = self._load_image(os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id))
                output_record['rgb_rear'] = self._load_image(os.path.join(route_dir, "rgb_rear", "%04d.jpg" % frame_id))
                if agent != 'rsu':
                    BEV = self._load_image(os.path.join(route_dir, "birdview", "%04d.jpg" % frame_id))
            elif extra_source is not None:
                lidar = extra_source['lidar']
                output_record['rgb_front'] = extra_source['rgb_front']
                output_record['rgb_left'] = extra_source['rgb_left']
                output_record['rgb_right'] = extra_source['rgb_right']
                output_record['rgb_rear'] = extra_source['rgb_rear']
                BEV = None # extra_source['bev']

            # if agent == 'ego':
            #     lidar = lidar*0

            output_record['lidar_np'] = lidar
            lidar_transformed = np.zeros((output_record['lidar_np'].shape))
            lidar_transformed[:,0] = output_record['lidar_np'][:,1]
            lidar_transformed[:,1] = -output_record['lidar_np'][:,0]
            lidar_transformed[:,2:] = output_record['lidar_np'][:,2:]
            output_record['lidar_np'] = lidar_transformed.astype(np.float32)
            output_record['lidar_np'][:, 2] += measurements['lidar_pose_z']

        if visible_actors is not None:
            actors_data = self.filter_actors_data_according_to_visible(actors_data, visible_actors)

        if agent == 'rsu' :
            for actor_id in actors_data.keys():
                if actors_data[actor_id]['tpe'] == 0:
                    box = actors_data[actor_id]['box']
                    if abs(box[0]-0.8214) < 0.01 and abs(box[1]-0.18625) < 0.01 :
                        actors_data[actor_id]['tpe'] = 3

        output_record['params']['vehicles'] = {}
        for actor_id in actors_data.keys():

            ######################
            ## debug
            ######################
            # if agent == 'ego':
            #     continue

            if tpe in [0, 1, 3]:
                if actors_data[actor_id]['tpe'] != tpe:
                    continue

            # if actors_data[actor_id]['tpe'] != 0:
            #     continue

            if not ('box' in actors_data[actor_id].keys() and 'ori' in actors_data[actor_id].keys() and 'loc' in actors_data[actor_id].keys()):
                continue
            output_record['params']['vehicles'][actor_id] = {}
            output_record['params']['vehicles'][actor_id]['tpe'] = actors_data[actor_id]['tpe']
            yaw = math.degrees(math.atan(actors_data[actor_id]['ori'][1]/actors_data[actor_id]['ori'][0]))
            pitch = math.degrees(math.asin(actors_data[actor_id]['ori'][2]))
            output_record['params']['vehicles'][actor_id]['angle'] = [0,yaw,pitch]
            output_record['params']['vehicles'][actor_id]['center'] = [0,0,actors_data[actor_id]['box'][2]]
            output_record['params']['vehicles'][actor_id]['extent'] = actors_data[actor_id]['box']
            output_record['params']['vehicles'][actor_id]['location'] = [actors_data[actor_id]['loc'][0],actors_data[actor_id]['loc'][1],0]
            output_record['params']['vehicles'][actor_id]['speed'] = 3.6 * math.sqrt(actors_data[actor_id]['vel'][0]**2+actors_data[actor_id]['vel'][1]**2 )



        direction_list = ['front','left','right','rear']
        theta_list = [0,-60,60,180]
        camera_data_list = []
        for i, direction in enumerate(direction_list):
            if 'rgb_{}'.format(direction) in output_record:
                camera_data_list.append(output_record['rgb_{}'.format(direction)])
            output_record['params']['camera{}'.format(i)]['cords'] = \
                                                                    [measurements['x'] + 1.3*np.sin(measurements['theta']), measurements['y'] - 1.3*np.cos(measurements['theta']), 2.3,\
                                                                    0,measurements['theta']/np.pi*180 + theta_list[i],0]
            output_record['params']['camera{}'.format(i)]['extrinsic'] = measurements['camera_{}_extrinsics'.format(direction_list[i])]
            output_record['params']['camera{}'.format(i)]['intrinsic'] = measurements['camera_{}_intrinsics'.format(direction_list[i])]

        output_record['camera_data'] = camera_data_list

        bev_visibility_np = 255*np.ones((256,256,3), dtype=np.uint8)
        output_record['bev_visibility.png'] = bev_visibility_np


        if agent != 'rsu':
            output_record['BEV'] = BEV
        else:
            output_record['BEV'] = None

        return output_record

    def filter_actors_data_according_to_visible(self, actors_data, visible_actors):
        to_del_id = []
        for actors_id in actors_data.keys():
            if actors_id in visible_actors:
                continue
            to_del_id.append(actors_id)
        for actors_id in to_del_id:
            del actors_data[actors_id]
        return actors_data

    def get_visible_actors_one_term(self, route_dir, frame_id):
        cur_visible_actors = []
        measurements = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % frame_id))
        actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % frame_id))

        for actors_id in actors_data:
            if actors_data[actors_id]['tpe']==2:
                continue
            if not 'lidar_visible' in actors_data[actors_id]:
                cur_visible_actors.append(actors_id) # if it has not visible key, 
                print('Lose of lidar_visible!')
                continue
            if actors_data[actors_id]['lidar_visible']==1:
                cur_visible_actors.append(actors_id)
        return cur_visible_actors

    def get_visible_actors(self, scene_dict, frame_id):
        list_2 = []
        visible_actors = {} # id only
        visible_actors['car_0'] = self.get_visible_actors_one_term(scene_dict['ego'], frame_id)
        list_2 += self.get_visible_actors_one_term(scene_dict['ego'], frame_id)
        if self.fusion_mode != 'none':
            for i, route_dir in enumerate(scene_dict['other_egos']):
                visible_actors['car_{}'.format(i+1)] = self.get_visible_actors_one_term(route_dir, frame_id)
                list_2 += self.get_visible_actors_one_term(route_dir, frame_id)
            for rsu_dir in scene_dict['rsu']:
                visible_actors['rsu_{}'.format(i)] = self.get_visible_actors_one_term(rsu_dir, frame_id)
                list_2 += self.get_visible_actors_one_term(route_dir, frame_id)

        for keys in visible_actors:
            visible_actors[keys] = list(set(visible_actors[keys]))
        # return list(set(visible_actors))
        return visible_actors, list(set(list_2))

    def retrieve_base_data(self, idx, tpe='all', extra_source=None):
        if extra_source is None:
            scene_dict, frame_id = self.route_frames[idx]
            visible_actors = None
            visible_actors, _ = self.get_visible_actors(scene_dict, frame_id)

            data = OrderedDict()

            data['car_0'] = self.get_one_record(scene_dict['ego'], frame_id , agent='ego', visible_actors=visible_actors['car_0'], tpe=tpe)# visible_actors['car_0'])


            if self.params['train_params']['max_cav'] > 1:
                for i, route_dir in enumerate(scene_dict['other_egos']):
                    try: # 
                        data['car_{}'.format(i+1)] = self.get_one_record(route_dir, frame_id , agent='other_ego', visible_actors=visible_actors['car_{}'.format(i+1)], tpe=tpe)#visible_actors['car_{}'.format(i+1)]) #list2)# agent='other_ego'
                    except:
                        print('load other ego failed')
                        continue
                for i, rsu_dir in enumerate(scene_dict['rsu']):
                    try: #   data['rsu_{}'.format(i)] = self.get_one_record(rsu_dir, frame_id, agent='rsu', visible_actors=visible_actors['rsu_{}'.format(i)], tpe=tpe)
                        data['rsu_{}'.format(i)] = self.get_one_record(rsu_dir, frame_id, agent='rsu', visible_actors=visible_actors['rsu_{}'.format(i)], tpe=tpe) #visible_actors['rsu_{}'.format(i)]) #list2)  
                    except:
                        print('load rsu failed')
                        continue
        else:
            data = OrderedDict()
            scene_dict = None
            frame_id = None
            data['car_0'] = self.get_one_record(route_dir=None, frame_id=None , agent='ego', visible_actors=None, tpe=tpe, extra_source=extra_source['car_data'][0])# visible_actors['car_0'])

            if self.params['train_params']['max_cav'] > 1:
                if len(extra_source['car_data']) > 1:
                    for i in range(len(extra_source['car_data'])-1):
                        data['car_{}'.format(i+1)] = self.get_one_record(route_dir=None, frame_id=None , agent='other_ego', visible_actors=None, tpe=tpe, extra_source=extra_source['car_data'][i+1])

                for i in range(len(extra_source['rsu_data'])):
                    data['rsu_{}'.format(i)] = self.get_one_record(route_dir=None, frame_id=None , agent='rsu', visible_actors=None, tpe=tpe, extra_source=extra_source['rsu_data'][i])            
        
        data['car_0']['scene_dict'] = scene_dict
        data['car_0']['frame_id'] = frame_id
        
        return data


    def __len__(self):
        return len(self.route_frames)

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth" 

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]


    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self, 
                                cav_contents, 
                                reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.
        
        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
            np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_to_lidar, camera_intrinsic