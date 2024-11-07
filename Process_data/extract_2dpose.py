import numpy as np
import os
x_train = np.load('./data/train_joint.npy',mmap_mode='r')
y_train = np.load('./data/train_label.npy')

x_valid = np.load('./data/val_joint.npy',mmap_mode='r')
y_valid = np.load('./data/val_label.npy')

x_test = np.load('./data/test_joint.npy',mmap_mode='r')
y_test = np.load('./data/test_label.npy')


if not os.path.exists('./save_2d_pose'):
    os.makedirs('./save_2d_pose')
np.savez('./save_2d_pose/joint_xz.npz', x_train = x_train[:,[0,2],:,:,:] , y_train = y_train , 
         x_test = x_valid[:,[0,2],:,:,:], y_test = y_valid)

np.savez('./save_2d_pose/joint_xz_test.npz',x_test = x_test[:,[0,2],:,:,:], y_test = y_test)

print("finish save 2d pose data")

