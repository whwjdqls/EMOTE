import numpy as np
import time
import torch
import torch.utils.data as data
import sys
import json
sys.path.append('../')
from datasets.talkingheaddataset import TalkingHeadDataset

config_path = "/home/whwjdqls99/EMOTE/configs/EMOTE/RAVDESS_dataset_test.json"
config = json.load(open(config_path))
TalkingHeadDataset(config, split='val')
# a = np.random.rand(100000,50+6+128)
# b = np.random.rand(100000,50+6+128)
# c = np.random.rand(100000,50+6+128)
# np.save(arr=a,file='a.npy')
# np.save(arr=b,file='b.npy')
# np.save(arr=c,file='c.npy')
# d = np.concatenate((a,b,c),axis=1)
# np.save('debug.npy',d)

# np.savez_compressed('debug.npz',a=a,b=b,c=c)

# cur_time = time.time()
# dat = np.load('debug.npz')

# # print(dat.files)
# for i in dat.files:
#     dat[i]
# print("time for loading npz:", time.time()-cur_time)  

# cur_time = time.time()
# for i in dat.files:
#     np.load(f'{i}.npy')
# print("time for loading 3 npy:",time.time()-cur_time)

# cur_time = time.time()
# dat = np.load('debug.npy')
# print("time for loading npy:",time.time()-cur_time)

