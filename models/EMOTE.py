import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from .wav2vec import Wav2Vec2Model
import torch.nn.functional as F
import copy
import einops

class EMOTE(nn.Module) :
    def __init__(self, config) :
        super(EMOTE, self).__init__()
        # for audio encoder
        config = Wav2Vec2Config(num_hidden_layers=config['EMOTE_config']['num_hidden_layers'])
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", config=config)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_map = nn.Liinear(768,128)
        # self.audio_feature_map = nn.Linear(768, config['EMOTE_config']['feature_dim'], bias=False)
        self.styling_layer = nn.Linear(config['EMOTE_config']['condition_num'], 128)

    def encode_audio(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        output = self.audio_encoder(audio).last_hidden_state # (BS,50*sec,768) : (BS,20,768)
        output = self.audio_map(output) # (BS, T, 128)
        output = output.transpose(1,2) # (BS, 128, T)
        # output = self.audio_feature_map(output) # (BS,50*sec,256) : (BS,20,512)
        return output # (BS, 128, T)

    def encode_style(self, condition) :
        '''
        condition : (BS, condition_num)
        condition_num = emotion + intensity + actors = 53
        '''
        output = self.styling_layer(condition) # (BS, condition_num)
        output = output.transpose(0,1).unsqueeze(2) # (BS, 128, 1)
        return output

    def forward(self, audio, condition) :
        audio_embedding = encode_audio(audio) # (BS, 128, T)
        style_embedding = encode_style(condition) # (BS, 768, T)

        audio_style_sum = torch.cat([audio_embedding, style_embedding], dim=2)
