import os
import glob
import torch
import json
import numpy as np
import torch.utils.data as data
import tqdm
from .dataset_utils import get_FLAME_params_MEAD
from transformers import Wav2Vec2Processor
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


class TalkingHeadDataset(data.Dataset):
    def __init__(self, config, split='train'):
        # split
        self.split = split
        config = config['data'] 
        self.config = config
        self.dataset = config['dataset']
        # data path
        self.audio_dir = config["audio_dir"]
        self.expression_feature_dir = config["expression_dir"]
        # window size
        self.window_size = config["window_size"] # T in EMOTE paper
        self.start_clip = config["start_clip"]
        self.end_clip = config["end_clip"]
        self.stride = config["stride"]
        # list for features
        self.inputs = []
        self.labels = []
        # if args.use_SER_encoder :
        #     self.processor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
        # else : 
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        #open RAVDESS
        if self.dataset == 'RAVDESS':
            uid_path = "./RAVDESS_uid_v2.json"
            with open(uid_path) as f:
                uid_dict = json.load(f)
                
            uid_list = uid_dict[self.split]
            
            for uid in tqdm.tqdm(uid_list):
                actor_name = 'Actor_' + uid.split('-')[-1]
                actor_id = int(uid.split('-')[-1])
                emotion = int(uid.split('-')[2])
                intensity = int(uid.split('-')[3])
                if actor_id % 2 == 0: # if actor num is even
                    gender = 1 # actor is female
                else: # if actor num is odd
                    gender = 0 # actor is male
                
                # for compatibility with MEAD
                audio_dir = self.audio_dir
                if "MEAD" in audio_dir:
                    audio_dir = audio_dir.replace("MEAD", "RAVDESS")
                audio_path = os.path.join(audio_dir, actor_name , uid + '.npy')

                audio_samples = np.load(audio_path)
                audio_samples_lib = torch.tensor(audio_samples, dtype=torch.float32)
                audio_samples = torch.squeeze(self.processor(audio_samples_lib, sampling_rate=16000, return_tensors="pt").input_values) 
                
                # for compatibility with MEAD
                expression_feature_dir = self.expression_feature_dir
                if "MEAD" in expression_feature_dir:
                    expression_feature_dir = expression_feature_dir.replace("MEAD", "RAVDESS")
                expression_feature_path = os.path.join(expression_feature_dir, actor_name, uid + '.json')
                param_dict = get_FLAME_params_RAVDESS(expression_feature_path, args.with_jaw, args.with_shape)
                expression_feature = torch.tensor(param_dict['expression'], dtype=torch.float32) #(len, 100)
 
                # generate input samples by slicing
                # JB - changed clip_length so t
                audio_start, audio_end, audio_stride, audio_window =  np.array([self.start_clip, self.end_clip, self.stride, self.window_size] )* 1600
                exp_start, exp_end, exp_stride, exp_window = np.array([self.start_clip, self.end_clip, self.stride, self.window_size]) * 3
            
                # if split == 'val': # for validation, always clip for 1.2 seconds & stride becomes 0.2 seconds
                #     audio_start, audio_end, audio_stride,audio_window ,\
                #         exp_start,exp_end ,exp_stride, exp_window =  [0]*8
                if split == 'val':
                    self.inputs.append([audio_samples_slice, expression_samples_slice])
                    self.labels.append([emotion, intensity, gender, RAVDESS_ACTOR_DICT[actor_id]])
                else:
                    for audio_start, expression_start in zip(
                        range(audio_start, audio_samples.shape[0] - audio_window  - audio_end, audio_stride),
                        range(exp_start, expression_feature.shape[0] - exp_window  - exp_end, exp_stride)
                    ):
                        audio_samples_slice = audio_samples[audio_start:audio_start+audio_window]
                        expression_samples_slice = expression_feature[exp_start:exp_start+exp_window]
                        # check the length -> if original length is smaller than feature len, pass
                        if audio_samples_slice.shape[0] != audio_window or expression_samples_slice.shape[0] != exp_window:
                            continue
                        # JB 
                        self.inputs.append([audio_samples_slice, expression_samples_slice])
                        self.labels.append([emotion, intensity, gender, RAVDESS_ACTOR_DICT[actor_id]])
                        # [int, int, int, int]
        if self.dataset == 'MEAD':
            # pass
            # uid_path = "../MEAD_uid.json"
            uid_path = './MEAD_uid.json'
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

                emotion = EMOTION_DICT[uid.split('_')[1]]
                intensity = int(uid.split("_")[3]) #level_1
                gender = GENDER_DICT[uid.split('_')[0][0]] # M -> 0, W -> 1
                expression_feature_path = os.path.join(self.expression_feature_dir, actor_name, uid + '.json')
                param_dict = get_FLAME_params_MEAD(expression_feature_path,True,False)
                

                # generate input samples by slicing
                # JB - changed clip_length so t


                # JB - changed clip_length so t
                audio_start, audio_end, audio_stride, audio_window =  np.array([self.start_clip, self.end_clip, self.stride, self.window_size]) * 1600
                exp_start, exp_end, exp_stride, exp_window = np.array([self.start_clip, self.end_clip, self.stride, self.window_size])* 3
                
                episodes = param_dict.keys()
                for episode in episodes :
                    expression_feature = torch.tensor(param_dict[episode]['expression'], dtype=torch.float32) #(len, 100)

                    episode_name = uid + '_' + episode
                    
                    if episode_name in MEAD_episode_exceptions:
                        continue
                    
                    audio_path = os.path.join(self.audio_dir, actor_name , episode_name + '.npy')
                    audio_samples = np.load(audio_path)
                    audio_samples_lib = torch.tensor(audio_samples, dtype=torch.float32)
                    audio_samples = torch.squeeze(self.processor(audio_samples_lib, sampling_rate=16000, return_tensors="pt").input_values)
                    if split == 'val':
                        self.inputs.append([audio_samples, expression_feature])
                        self.labels.append([emotion, intensity, gender, actor_id])
                    else:
                        for audio_start, expression_start in zip(
                            range(audio_start, audio_samples.shape[0] - audio_window  - audio_end, audio_stride),
                            range(exp_start, expression_feature.shape[0] - exp_window  - exp_end, exp_stride)
                        ):
                            audio_samples_slice = audio_samples[audio_start:audio_start+audio_window]
                            expression_samples_slice = expression_feature[exp_start:exp_start+exp_window]
                            # check the length -> if original length is smaller than feature len, pass
                            if audio_samples_slice.shape[0] != audio_window or expression_samples_slice.shape[0] != exp_window:
                                continue
                            # JB 
                            self.inputs.append([audio_samples_slice, expression_samples_slice])
                            self.labels.append([emotion, intensity, gender, actor_id])
                        # [int, int, int, int]
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
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