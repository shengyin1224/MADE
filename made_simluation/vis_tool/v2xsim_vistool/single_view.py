
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import simple_plot3d.canvas_3d as canvas_3d
import numpy as np
import os
import copy
from pyquaternion import Quaternion

from simple_dataset import SimpleDataset


COLOR = ['red','springgreen','dodgerblue', 'darkviolet', 'orange']
COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
COLOR_PC = [tuple([int(cc*0.2 + 255*0.8) for cc in c]) for c in COLOR_RGB]
CLASSES = ['agent1', 'agent2', 'agent3', 'agent4', 'agent5']

canvas_shape=(800, 1200)
camera_center_coords=(-10, 0, 10)
camera_focus_coords=(-10 + 0.5396926, 0, 10 - 0.34202014)
left_hand = False


def generate_object_corners_v2x(cav_contents,
                               reference_lidar_pose):
        """
        v2x-sim dataset

        Retrieve all objects (gt boxes)

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            In fact, only the ego vehile needs to generate object center

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (n, 8, 3).
        object_ids : list
            Length is number of bbx in current sample.
        """
        # from opencood.data_utils.datasets import GT_RANGE
        max_num = 200
        gt_boxes = cav_contents[0]['params']['vehicles'] # notice [N,10], 10 includes [x,y,z,dx,dy,dz,w,a,b,c]
        object_ids = cav_contents[0]['params']['object_ids']
        
        object_dict = {"gt_boxes": gt_boxes, "object_ids":object_ids}

        output_dict = {}
        lidar_range = (-64,-64,-3,64,64,2)
        x_min, y_min, z_min, x_max, y_max, z_max = lidar_range
        
        gt_boxes = object_dict['gt_boxes']
        object_ids = object_dict['object_ids']
        for i, object_content in enumerate(gt_boxes):
            x,y,z,dx,dy,dz,w,a,b,c = object_content

            q = Quaternion([w,a,b,c])
            T_world_object = q.transformation_matrix
            T_world_object[:3,3] = object_content[:3]

            T_world_lidar = reference_lidar_pose

            object2lidar = np.linalg.solve(T_world_lidar, T_world_object) # T_lidar_object

            # shape (3, 8)
            x_corners = dx / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1]) # (8,)
            y_corners = dy / 2 * np.array([-1,  1,  1, -1, -1,  1,  1, -1])
            z_corners = dz / 2 * np.array([-1, -1, -1, -1,  1,  1,  1,  1])

            bbx = np.vstack((x_corners, y_corners, z_corners)) # (3, 8)

            # bounding box under ego coordinate shape (4, 8)
            bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

            # project the 8 corners to world coordinate
            bbx_lidar = np.dot(object2lidar, bbx).T # (8, 4)
            bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0) # (1, 8, 3)

            bbox_corner = copy.deepcopy(bbx_lidar)

            center = np.mean(bbox_corner, axis=1)[0]

            if (center[0] > x_min and center[0] < x_max and 
               center[1] > y_min and center[1] < y_max and 
               center[2] > z_min and center[2] < z_max) or i==3:
                output_dict.update({object_ids[i]: bbox_corner})

        object_np = np.zeros((max_num, 8, 3))
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            object_ids.append(object_id)

        # should not appear repeated items
        object_np = object_np[:len(object_ids)]

        return object_np, object_ids


def main():
    ## basic setting
    dataset = SimpleDataset(root_dir='./v2xsim1.0_info/v2xsim_infos_vis[31].pkl')
    import pdb; pdb.set_trace()
    data_dict_demo = dataset[0]
    cav_ids = list(data_dict_demo.keys())
    cav_invert_dict = dict() # cav_id -> 0/1/2
    for (idx, cav_id) in enumerate(cav_ids):
        cav_invert_dict[cav_id] = idx

    recs = []
    for i in range(0,len(cav_ids)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=COLOR[i]))

    ## matplotlib setting
    plt.figure()
    plt.style.use('dark_background')


    ## draw
    print("loop over dataset")
    dataset_len = len(dataset)
    for idx in range(dataset_len):
        print(idx)
        base_data_dict = dataset[idx]
        
        # retrieve all bbox, under world coordinate
        for cav_id, cav_content in base_data_dict.items():
            lidar_np_ego_agg = np.zeros((0, 4))
            ego_pose = cav_content['params']['lidar_pose']

            cav_contents = [base_data_dict[1]]
            object_np, object_ids = generate_object_corners_v2x(cav_contents, ego_pose)

            lidar_np_ego_agg = cav_content['lidar_np']

            canvas = canvas_3d.Canvas_3D(canvas_shape, camera_center_coords, camera_focus_coords, left_hand=left_hand) 

            canvas_xy, valid_mask = canvas.get_canvas_coords(lidar_np_ego_agg)
            canvas.draw_canvas_points(canvas_xy[valid_mask], colors=COLOR_PC[cav_invert_dict[cav_id]])
            canvas.draw_boxes(object_np, colors=COLOR_RGB[cav_invert_dict[cav_id]])

            plt.axis("off")
            plt.imshow(canvas.canvas)
            plt.tight_layout()

            save_path = f"./result_v2x/single_view_{CLASSES[cav_invert_dict[cav_id]]}"

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            plt.savefig(f"{save_path}/{idx:02d}.png", transparent=False, dpi=300)
            plt.clf()

if __name__ == "__main__":
    main()