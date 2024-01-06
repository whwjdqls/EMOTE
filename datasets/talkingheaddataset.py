import os
import glob
import torch
import json
import numpy as np
import torch.utils.data as data
import tqdm
# from .dataset_utils import get_FLAME_params_RAVDESS, get_FLAME_params_MEAD
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
import librosa
import time

# (10-25) 
RAVDESS_ACTOR_DICT = {1 : 0, 3 : 1, 4 : 2, 5 : 3, 6 : 4, 7 : 5, 8 : 6, 9 : 7, 10 : 8, 11 : 9, 12 : 10, 13 : 11, 14 : 12, 15 : 13, 16 : 14, 17 : 15, 18 : 16, 19 : 17, 20 : 18, 21 : 19, 22 : 20, 23 : 21, 24 : 22, 25 : 23, # for train
                      2 : 24} # for val

# (12-25) JB - following EMOTE 
training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                ] # 32 ids
val_ids = ['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036']  # 7 ids

test_ids = ['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040'] # 8 ids

                    # 32 train_ids
MEAD_ACTOR_DICT = {'M003': 0, 'M005': 1, 'M007': 2, 'M009': 3, 'M011': 4, 'M012': 5, 'M013': 6, 'M019': 7, 'M022': 8, 'M023': 9, 'M024': 10, 'M025': 11, 'M026': 12, 'M027': 13, 'M028': 14, 'M029': 15, 'M030': 16, 'M031': 17, 'W009': 18, 'W011': 19, 'W014': 20, 'W015': 21, 'W016': 22, 'W018': 23, 'W019': 24, 'W021': 25, 'W023': 26, 'W024': 27, 'W025': 28, 'W026': 29, 'W028': 30, 'W029': 31, 
                   'M032': 32, 'M033': 33, 'M034': 34, 'M035': 35, 'W033': 36, 'W035': 37, 'W036': 38, # 7 val_ids
                   'M037': 39, 'M039': 40, 'M040': 41, 'M041': 42, 'M042': 43, 'W037': 44, 'W038': 45, 'W040': 46} # 8 test_ids

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
            uid_path = '/home/whwjdqls99/EMOTE/datasets/MEAD_uid_test.json'
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
                    expression_feature = torch.tensor(param_dict[episode]['expression'], dtype=torch.float32) #(len, 50)
                    jaw_feature = torch.tensor(param_dict[episode]['jaw'], dtype=torch.float32) #(len, 3)
                    expression_feature = torch.cat([expression_feature, jaw_feature], dim=1) #(len, 53)
                    
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
    
class TalkingHeadDataset_new(data.Dataset):
    def __init__(self, config, full_length=False, split='train'):
        # split
        self.split = split
        self.full_length = full_length
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
            # pass # (12-25) change to using glob glob to get all files
            actor_list = []
            if self.split == 'train':
                actor_list = training_ids
            elif self.split == 'val':
                actor_list = val_ids
            elif self.split == 'test':
                actor_list = test_ids
            elif self.split == 'debug':
                actor_list = ['M003']
            elif self.split == 'visualize':
                actor_list = ['M032', 'M033']
            elif self.split in training_ids + val_ids + test_ids: # for single actor
                actor_list = [self.split]
            else:
                raise NotImplementedError('split should be train, val, test, debug, visualize or single actor id')
            print(f'making dataset with {len(actor_list)} actors {actor_list}')
            all_actor_list = os.listdir(self.expression_feature_dir)
            # check if all the actors are in the directory
            if set(actor_list).intersection(set(all_actor_list)) != set(actor_list):
                print(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
                print(f'missing actors : {set(actor_list).difference(set(all_actor_list))}')
                print(f'using only {set(actor_list).intersection(set(all_actor_list))}')
                actor_list = set(actor_list).intersection(set(all_actor_list))
                # raise ValueError(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
            
            audio_start, audio_end, audio_stride, audio_window =  np.array([self.start_clip, self.end_clip, self.stride, self.window_size]) * 1600
            exp_start, exp_end, exp_stride, exp_window = np.array([self.start_clip, self.end_clip, self.stride, self.window_size])* 3
            
            file_paths = []
            for actor in actor_list:
                file_paths += glob.glob(os.path.join(self.expression_feature_dir, actor, '*.npy'))
            
            for file_path in tqdm.tqdm(file_paths):
                uid = file_path.split('/')[-1].split('.')[0]
                
                actor_name = uid.split('_')[0] # M005
                actor_id = MEAD_ACTOR_DICT[actor_name] # name -> id
                emotion = int(uid.split('_')[1])    
                intensity = int(uid.split("_")[2])
                gender = GENDER_DICT[uid.split('_')[0][0]] # M -> 0, W -> 1
                
                param_dict = {}
                parameters = np.load(file_path)
                try :
                    param_dict['expression'] = parameters[:,:50]
                    param_dict['jaw'] = parameters[:,50:53]
                    param_dict['shape'] = parameters[:,53:]
                except :
                    print(f'Something wrong with {file_path}')
                    print(f'expression shape : {parameters.shape}')
                    continue

                expression_feature = torch.tensor(param_dict['expression'], dtype=torch.float32) #(len, 50)
                jaw_feature = torch.tensor(param_dict['jaw'], dtype=torch.float32) #(len,3)
                param_feature = torch.cat([expression_feature, jaw_feature], dim=1)

                audio_path = os.path.join(self.audio_dir, actor_name , uid + '.npy')

                if not os.path.exists(audio_path) :
                    print(f'{audio_path} doesnt exist')
                    continue

                audio_samples = np.load(audio_path)
                audio_samples = torch.tensor(audio_samples, dtype=torch.float32)
                audio_samples = torch.squeeze(self.processor(audio_samples, sampling_rate=16000, return_tensors="pt").input_values)
                
                if self.full_length:
                    # full length audio
                    self.inputs.append([audio_samples, param_feature])
                    self.labels.append([emotion, intensity, gender, actor_id])
                    continue
                
                for audio_start_, expression_start_ in zip(
                    range(audio_start, audio_samples.shape[0] - audio_window - audio_end, audio_stride),
                    range(exp_start, param_feature.shape[0] - exp_window - exp_end, exp_stride)
                ):
                    audio_samples_slice = audio_samples[audio_start_:audio_start_+audio_window]
                    
                    expression_samples_slice = param_feature[expression_start_:expression_start_+exp_window]
                    
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
    
    
# if __name__ == "__main__":
#     import json
    
#     config_path = "/home/whwjdqls99/EMOTE/configs/FLINT/FLINT_V1.json"
#     config = json.load(open(config_path))
#     data = FlameDataset(config)
#     print(data[0].shape)
#     print(len(data))
    
#     config_path = "/home/whwjdqls99/EMOTE/configs/EMOTE/EMOTE_V1.json"
#     config = json.load(open(config_path))
#     data = TalkingHeadDataset(config, split='train')
#     data_label = data[0]
#     data, label = data_label
#     flame_param, audio = data
#     print(flame_param.shape)
#     print(audio.shape)
#     print(audio.dtype)
#     print(flame_param.dtype)
    
    