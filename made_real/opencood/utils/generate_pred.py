import numpy as np
import datetime
import os

def generate_pred(path, save_path, agent_num_list = [], attack_gt = 'pred', range_data = []):

    pred = []

    for num in range_data:
        tmp_num = agent_num_list[num]
        if tmp_num - 1 == 0:
            continue
        else:
            sample = np.load(path + f'/sample_{num}.npy', allow_pickle=True)
            delete_agent_num = len(sample)
            for i in range(agent_num_list[num] - 1):
                if i + 1 in sample:
                    pred.append(1)
                else:
                    pred.append(0)

    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%m-%d %H:%M')
    save_path_final = save_path + '/' + attack_gt + '/' + 'multi_test' 

    # 判断是否创建过文件夹
    if not os.path.exists(save_path_final):
        os.makedirs(save_path_final)

    np.save(save_path_final + '/' + time_str  + '- pred.npy', pred)

    return pred