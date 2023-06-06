import skvideo.io  
import os

root = "/Users/melkor/Desktop/datasets/PHASE/train/"
path = root + "D082120_00440100_0_F8_E12_G['LMO', 1, 3, 1]_['LMO', 1, 3, 1]_ST1_1_SZ0_0_3_1_P4_13_17_14_A0_0_C0_0_AN3.76_5.79_MCTS_L1_R0.0_0.0_PL1_EL1_0_0_s1000_r10_cI1.25_cB1000_e9.mp4"

import numpy as np
import matplotlib.pyplot as plt

scale = 2

train_size = 8 * scale
test_size  = 2 * scale

train_itr = 0
test_itr = 0

import os
for root, dirs, files in os.walk(root, topdown=False):
    for name in files:
        root = "/Users/melkor/Desktop/datasets/PHASE/train/"
        path = os.path.join(root, name)
        if path[-3:] == "mp4":    
            videodata = skvideo.io.vread(path)  
            max_size = videodata.shape[0] - 1
            for i in range(train_size):
                idx = np.random.random_integers(0,max_size)
                im = videodata[idx]
                root = "/Users/melkor/Desktop/datasets/static_phase"
                plt.imsave(root+"/train"+"/{}.png".format(train_itr),im)
                train_itr += 1
            for i in range(test_size):
                idx = np.random.random_integers(0,max_size)
                im = videodata[idx]
                root = "/Users/melkor/Desktop/datasets/static_phase"
                plt.imsave(root+"/test"+"/{}.png".format(test_itr),im)
                test_itr += 1
