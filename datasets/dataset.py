import os
import glob
import torch
import numpy as np
import torch.utils.data as data
import cv2
# dataset to store whole clip

class FlameDataset(data.Dataset):
    def __init__(self, config):
        # list for all data
        self.inputs = []
        
        # base data directory
        self.data_dir = config["data"]["data_dir"]
        self.window_size = config["data"]["window_size"] # T in EMOTE paper
        
        # get all file paths
        self.file_paths = glob.glob(os.path.join(self.data_dir, '*/*.npy'))
        
        # load all data
        for file_path in self.file_paths:
            params = np.load(file_path)
            for i in range(len(params) - self.window_size):
                self.inputs.append(params[i:i+self.window_size]) # params.shape = (T, 53)
        
        
    def __getitem__(self,index):
        return self.inputs[index]
    
    def __len__(self):
        return len(self.inputs)
    
    
    
if __name__ == "__main__":
    import json
    
    config_path = "/home/whwjdqls99/EMOTE/configs/FLINT/FLINT_V1.json"
    config = json.load(open(config_path))
    data = FlameDataset(config)
    print(data[0].shape)
    print(len(data))
    
    