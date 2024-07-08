import cv2
import numpy as np
import glob
import os

projnames = os.listdir("./result_v2x")
print(projnames)

for projname in projnames:
    img_array = []
    for filename in sorted(glob.glob(f'./result_v2x/{projname}/*.png')):
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter(f'./result_v2x_video/{projname}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 9, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()