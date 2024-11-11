# import numpy as np


# data = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/save_3d_pose/V2_joint_train.npz',allow_pickle=True)['x_train']
# print(data.shape)
# print(type(data))

# lst = []
# for i in range(100):
#     lst.append(data[i].numpy())

# np.savetxt('output.txt',  np.array(lst))


import numpy as np

np.savez('/data2/songxinshuai/ICMEW2024-Track10/Process_data/save_3d_pose/joint3d.npz', 
         x_train=np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/train_joint.npy')
         ,y_train=np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/train_label.npy'),
         x_test = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/test_A_joint.npy'),
         y_test = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/test_A_label.npy'))


# python main.py --config /data2/songxinshuai/ICMEW2024-Track10/Model_inference/Mix_GCN/config/msg3d_joint.yaml --phase train --save-score True  --device 0 
