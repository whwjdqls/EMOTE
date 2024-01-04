import os
import glob
import torch
import json
import numpy as np
import torch.utils.data as data
import tqdm
from .dataset_utils import get_FLAME_params_MEAD
# import cv2
# dataset to store whole clip
# (10-18) exceptions on actor level
MEAD_actor_exceptions = ["W017", "W014", "W032"]
# (10-18) exceptions on uid level 
MEAD_uid_exceptions= ["M022_surprised_level_3"  , "W019_surprised_level_3" , "W033_sad_level_1", "W037_happy_level_3"]
# (10-18) exceptions on episode level (due to missing audio in mp4 files)
MEAD_episode_exceptions = ['M041_sad_level_2_020','M041_sad_level_2_021','M041_sad_level_2_022','M041_sad_level_2_023']
# (10-25) 
RAVDESS_ACTOR_DICT = {1 : 0, 3 : 1, 4 : 2, 5 : 3, 6 : 4, 7 : 5, 8 : 6, 9 : 7, 10 : 8, 11 : 9, 12 : 10, 13 : 11, 14 : 12, 15 : 13, 16 : 14, 17 : 15, 18 : 16, 19 : 17, 20 : 18, 21 : 19, 22 : 20, 23 : 21, 24 : 22, 25 : 23, # for train
                      2 : 24} # for val
MEAD_ACTOR_DICT = {'M005': 0, 'M007': 1, 'M009': 2, 'M011': 3, 'M012': 4, 'M013': 5, 'M019': 6,  'M023': 8, 'M024': 9, 'M025': 10, 'M026': 11, 'M027': 12, 'M028': 13, 'M029': 14, 'M030': 15, 'M031': 16, 'M033': 17, 'M034': 18, 'M035': 19, 'M037': 20, 'M039': 21, 'M040': 22, 'M041': 23, 'M042': 24,  'W015': 26, 'W016': 27, 'W018': 28, 'W019': 29, 'W021': 30, 'W023': 31, 'W024': 32, 'W025': 33, 'W026': 34, 'W028': 35, 'W029': 36, 'W033': 37, 'W035': 38, 'W036': 39, 'W037': 40, 'W038': 41, 'W040': 42,
                   'M022': 7, 'W011': 25, 'M003' : 43, 'W009' : 44} # for val
EMOTION_DICT = {'neutral': 1, 'calm': 2, 'happy': 3, 'sad': 4, 'angry' :  5, 'fear': 6, 'disgusted': 7, 'surprised': 8, 'contempt' : 9}
GENDER_DICT = {'M' : 0, 'W' : 1}

class MotionPriorMEADDataset(data.Dataset):
    def __init__(self, config, split='train'):
        # list for all data
        config = config['data']
        self.config = config
        self.inputs = []
        self.split = split
        # base data directory
        self.data_dir = config["data_dir"]
        self.window_size = config["window_size"] # T in EMOTE paper
        uid_path = '/home/whwjdqls99/Dynamic-Emotion-Embedding/DEE/MEAD_uid.json'
        with open(uid_path) as f:
            uid_dict = json.load(f)
            
        uid_list = uid_dict[self.split]
        for uid in tqdm.tqdm(uid_list):
            if uid in MEAD_uid_exceptions:
                continue
            
            actor_name = uid.split('_')[0] #M005
            actor_id = MEAD_ACTOR_DICT[actor_name]
            if actor_name in MEAD_actor_exceptions:
                continue

            emotion = EMOTION_DICT[uid.split('_')[1]] # angry -> 5
            intensity = int(uid.split("_")[3]) # level_1 -> 1
            gender = GENDER_DICT[uid.split('_')[0][0]] # M -> 0, W -> 1
            
            expression_feature_path = os.path.join(self.data_dir, actor_name, uid + '.json')
            param_dict = get_FLAME_params_MEAD(expression_feature_path, True, False)
            
            episodes = param_dict.keys()
            for episode in episodes :
                expression_feature = torch.tensor(param_dict[episode]['expression'], dtype=torch.float32) #(len, 50)
                jaw_feature = torch.tensor(param_dict[episode]['jaw'], dtype=torch.float32) #(len, 3)
                # print(jaw_feature)
                # if args.smooth_expression:
                #     expression_feature = savgol_filter(expression_feature, 5, 2, axis=0)
                #     jaw_feature = savgol_filter(jaw_feature, 5, 2, axis=0)
                    
                episode_name = uid + '_' + episode
                if episode_name in MEAD_episode_exceptions:
                    continue

                for expression_start in range(0, expression_feature.shape[0] - self.window_size, self.window_size):
                    expression_samples_slice = np.array(expression_feature[expression_start:expression_start+self.window_size])
                    jaw_samples_slice= np.array(jaw_feature[expression_start:expression_start+self.window_size])
                    params = np.concatenate((expression_samples_slice, jaw_samples_slice), axis=-1) # len, 53
                                           
                    self.inputs.append(params)


    def __getitem__(self,index):
        return self.inputs[index]
    
    def __len__(self):
        return len(self.inputs)
    
    
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
            params = np.load(file_path).astype(np.float32)
            for i in range(len(params) - self.window_size):
                self.inputs.append(params[i:i+self.window_size]) # params.shape = (T, 53)
 
        
    def __getitem__(self,index):
        return self.inputs[index]
    
    def __len__(self):
        return len(self.inputs)
    
    
if __name__ == "__main__":
    import json
    
    # config_path = "/home/whwjdqls99/EMOTE/configs/FLINT/FLINT_V1.json"
    # config = json.load(open(config_path))
    # data = FlameDataset(config)
    # print(data[0].shape)
    # print(len(data))
    
    config_path = "/home/whwjdqls99/EMOTE/configs/FLINT/FLINT_V1_MEAD.json"
    config = json.load(open(config_path))
    data = MEADDataset(config, split='val')
    print(data[0].shape)
    print(len(data))