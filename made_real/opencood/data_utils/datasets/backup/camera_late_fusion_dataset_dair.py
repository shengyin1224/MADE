from opencood.data_utils.datasets import camera_late_fusion_dataset
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.transformation_utils import rot_and_trans_to_trasnformation_matrix, x1_to_x2, x_to_world, tfm_to_pose, \
                        veh_side_rot_and_trans_to_trasnformation_matrix, inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from PIL import Image
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
    gen_dx_bx,
    load_camera_data,
    coord_3d_to_2d
)
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
import opencood.data_utils.post_processor as post_processor
import opencood.utils.pcd_utils as pcd_utils
import json
import os
import numpy as np
import torch
from collections import OrderedDict

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

class CameraLateFusionDatasetDAIR(camera_late_fusion_dataset.CameraLateFusionDataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]
        self.max_cav = 2

        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        self.anchor_box = self.post_processor.generate_anchor_box()
        self.anchor_box_torch = torch.from_numpy(self.anchor_box)

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']
        self.split_info = load_json(split_dir)
        co_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()

        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

        # depth gt
        self.use_gt_depth = True \
            if ('camera_params' in params and params['camera_params']['use_depth_gt']) \
            else False
        self.use_fg_mask = True \
            if ('use_fg_mask' in params['loss']['args'] and params['loss']['args']['use_fg_mask']) \
            else False
        self.supervise_single = True \
            if ('supervise_single' in params['train_params'] and params['train_params']['supervise_single']) \
            else False

        assert self.use_gt_depth is False # no ground truth depth
        assert self.supervise_single is False
        self.preload = False

        self.veh_only = True if ('veh_only' in params and params['veh_only']) \
            else False
        self.inf_only = True if ('inf_only' in params and params['inf_only']) \
            else False

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.

        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        data[0] = OrderedDict() # veh-side
        data[0]['ego'] = True
        data[1] = OrderedDict() # inf-side
        data[1]['ego'] = False

        # veh-side
        data[0]['params'] = OrderedDict()
        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file, novatel_to_world_json_file) # lidar to world
        
        vehicle_side_path = os.path.join(self.root_dir, 'vehicle-side/label/camera/{}.json'.format(veh_frame_id)) # use camera label? or lidar_backup
        data[0]['params']['vehicles'] = load_json(vehicle_side_path)

        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)
        data[0]['camera_data'] = load_camera_data([os.path.join(self.root_dir, frame_info["vehicle_image_path"])])
        data[0]['params']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                        load_json(os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/'+str(veh_frame_id)+'.json')))
        data[0]['params']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                        load_json(os.path.join(self.root_dir, 'vehicle-side/calib/camera_intrinsic/'+str(veh_frame_id)+'.json')))

        # inf-side
        data[1]['params'] = OrderedDict()
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")

        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file, system_error_offset)

        infra_side_path = os.path.join(self.root_dir, 'infrastructure-side/label/camera/{}.json'.format(inf_frame_id)) # use camera label? or virtuallidar
        data[1]['params']['vehicles'] = load_json(infra_side_path)

        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)
        data[1]['camera_data']= load_camera_data([os.path.join(self.root_dir,frame_info["infrastructure_image_path"])])
        data[1]['params']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                        load_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/'+str(inf_frame_id)+'.json')))
        data[1]['params']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                        load_json(os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/'+str(inf_frame_id)+'.json')))


        # if visualization
        if self.visualize:
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
            data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))

        if self.veh_only:
            data.pop(1)
        
        if self.inf_only:
            data.pop(0)
            data[1]['ego'] = True

        return data

    def __len__(self):
        return len(self.split_info)

    def get_item_single_car_camera(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.


        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
            including 'params', 'camera_data'

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # single label dict
        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
            [selected_cav_base], selected_cav_base["params"]["lidar_pose_clean"]
        )

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
            }
        )

        # generate targets label
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
        )
        selected_cav_processed.update({"label_dict": label_dict})


        # adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py
        camera_data_list = selected_cav_base["camera_data"]

        params = selected_cav_base["params"]
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        for idx, img in enumerate(camera_data_list):
            lidar_to_camera = params['extrinsic'].astype(np.float32) # R_cw
            camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
            camera_intrinsic = params['intrinsic'].astype(np.float32)

            intrin = torch.from_numpy(camera_intrinsic)
            rot = torch.from_numpy(
                camera_to_lidar[:3, :3]
            )  # R_wc, we consider world-coord is the lidar-coord
            tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            img_src = [img]

            # depth
            if self.use_gt_depth:
                depth_img = selected_cav_base["depth_data"][idx]
                img_src.append(depth_img)
            else:
                depth_img = None

            if self.use_fg_mask:
                _, _, fg_mask = coord_3d_to_2d(
                                box_utils.boxes_to_corners_3d(object_bbx_center[:len(object_ids)], self.params['postprocess']['order']),
                                camera_intrinsic,
                                camera_to_lidar,
                                image_H=1080, 
                                image_W=1920,
                                image=np.array(img),
                                idx=idx) 
                fg_mask = np.array(fg_mask*255, dtype=np.uint8)
                fg_mask = Image.fromarray(fg_mask)
                img_src.append(fg_mask)
            


            # data augmentation
            resize, resize_dims, crop, flip, rotate = sample_augmentation(
                self.data_aug_conf, self.train
            )
            img_src, post_rot2, post_tran2 = img_transform(
                img_src,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # decouple RGB and Depth

            img_src[0] = normalize_img(img_src[0])
            if self.use_gt_depth:
                img_src[1] = img_to_tensor(img_src[1]) * 255
            if self.use_fg_mask:
                img_src[-1] = img_to_tensor(img_src[-1])

            imgs.append(torch.cat(img_src, dim=0))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        selected_cav_processed.update(
            {
            "image_inputs": 
                {
                    "imgs": torch.stack(imgs), # [N, 3or4, H, W]
                    "intrins": torch.stack(intrins),
                    "rots": torch.stack(rots),
                    "trans": torch.stack(trans),
                    "post_rots": torch.stack(post_rots),
                    "post_trans": torch.stack(post_trans),
                }
            }
        )

        if self.visualize:
            lidar_np = selected_cav_base["lidar_np"]
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_points_by_range(
                lidar_np, self.params["preprocess"]["cav_lidar_range"]
            )
            # remove points that hit ego vehicle
            lidar_np = mask_ego_points(lidar_np)
            selected_cav_processed.update({"origin_lidar": lidar_np})

        return selected_cav_processed


    ### rewrite generate_object_center ###
    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):

        return self.post_processor.generate_object_center_dairv2x_single(cav_contents)

    def post_process(self, data_dict, output_dict):
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_no_fusion(self, data_dict, output_dict_ego, return_uncertainty=False):
        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict)

        if return_uncertainty:
            pred_box_tensor, pred_score, uncertainty = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
            return pred_box_tensor, pred_score, gt_box_tensor, uncertainty
        else:
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego)
            return pred_box_tensor, pred_score, gt_box_tensor