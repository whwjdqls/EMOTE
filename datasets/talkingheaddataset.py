import os
import glob
import torch
import json
import numpy as np
import torch.utils.data as data
import tqdm
from .dataset_utils import get_FLAME_params_MEAD
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor


def make_dataset_from_config(config, dataset_name):

class MEAD_TalkingHeadDataset(data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass


class RAVDESS_TalkingHeadDataset(data.Dataset):
    def __init__(self, config, split = 'val'):
        config = config['data']
        self.config = config # data config
        self.data_dir = config['data_dir']  # data directory path
        self.seq_len = config['seq_len']
        self.process_method = 'random'
        self.processed_dir = self.data_dir + '/processed'


        make_processed_data()
    

    def make_processed_data():
        if self.

        
