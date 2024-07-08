import cv2
import numpy as np
import glob
import os


scene_idx_list = [82, 86, 88]
agent_list = [0,1,2,3]
for scene_idx in scene_idx_list:

    for agent_idx in agent_list:

        eps1_path = f"experiments/visualize/N01_E1e-01_S10_vis/agent_{agent_idx}"
        eps05_path = f'experiments/visualize/N01_E5e-02_S10_vis/agent_{agent_idx}'
        robosac_path = f'experiments/visualize/robosac_N01_E2e-01_S10/agent_{agent_idx}'
        oracle_path = f'experiments/visualize/oracle_N01_E2e-01_S10/agent_{agent_idx}'

        # img_array = []
        # for idx in range(0,100):

        #     img_path = f"{path}/{scene_idx}_{idx}.png"
        #     if not os.path.exists(img_path):
        #         continue
        #     # import pdb; pdb.set_trace()
        #     img = cv2.imread(img_path)

        #     height, width, layers = img.shape
        #     size = (width,height)
        #     img_array.append(img)


        # out = cv2.VideoWriter(f'video/{path[-7:]}/video_3d_{scene_idx}_attack.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 9, size)
            
        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        # out.release()

        # img_array = []
        # for idx in range(0,100):

        #     img_path = f"{eps1_path}/{scene_idx}_{idx}.png"
        #     if not os.path.exists(img_path):
        #         continue
        #     # import pdb; pdb.set_trace()
        #     img = cv2.imread(img_path)
        #     height, width, layers = img.shape
        #     size = (width,height)
        #     img_array.append(img)

        # out = cv2.VideoWriter(f'video/{eps1_path[-7:]}/eps1_path_video_3d_{scene_idx}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 9, size)
            
        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        # out.release()

        # img_array = []
        # for idx in range(0,100):

        #     img_path = f"{eps05_path}/{scene_idx}_{idx}.png"
        #     if not os.path.exists(img_path):
        #         continue
        #     # import pdb; pdb.set_trace()
        #     img = cv2.imread(img_path)
        #     height, width, layers = img.shape
        #     size = (width,height)
        #     img_array.append(img)

        # out = cv2.VideoWriter(f'video/{eps05_path[-7:]}/eps05_path_video_3d_{scene_idx}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 9, size)
            
        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        # out.release()

        # img_array = []
        # for idx in range(0,100):

        #     img_path = f"{robosac_path}/{scene_idx}_{idx}.png"
        #     if not os.path.exists(img_path):
        #         continue
        #     # import pdb; pdb.set_trace()
        #     img = cv2.imread(img_path)
        #     height, width, layers = img.shape
        #     size = (width,height)
        #     img_array.append(img)

        # out = cv2.VideoWriter(f'video/{robosac_path[-7:]}/robosac_path_video_3d_{scene_idx}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 9, size)
            
        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        # out.release()

        img_array = []
        for idx in range(0,100):

            img_path = f"{oracle_path}/{scene_idx}_{idx}.png"
            if not os.path.exists(img_path):
                continue
            # import pdb; pdb.set_trace()
            img = cv2.imread(img_path)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        out = cv2.VideoWriter(f'video/{robosac_path[-7:]}/oracle_path_video_3d_{scene_idx}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 9, size)
            
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

# 图片组成视频