import numpy as np
import datetime
import os

def generate_npy(path, save_path, method = 'match', attack_gt = 'pred'):

    score = []
    label = []

    file_list = os.listdir(path)

    for file in file_list:

        if file.endswith('.npy'):
            file_path = os.path.join(path, file)
            sample = np.load(file_path, allow_pickle=True)
            agent_num = len(sample)
            for i in range(agent_num):
                agent = sample[i]
                if agent[1] == 0:
                    label.append(0)
                else:
                    label.append(1)
                score.append(agent[0][0])
            
    score = np.array(score)
    label = np.array(label)

    save_path_final = save_path + '/' + attack_gt + '/' + method 

    # 判断是否创建过文件夹
    if not os.path.exists(save_path_final):
        os.makedirs(save_path_final)

    np.save(save_path_final + '/' +  'score.npy', score)
    np.save(save_path_final + '/' +  'label.npy', label)

    return score, label