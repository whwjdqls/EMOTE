import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from .wav2vec import Wav2Vec2Encoder
import torch.nn.functional as F
import copy
import einops
# from .sequence_encoder import LinearSequenceEncoder, ConvSquasher, StackLinearSquash
# from inferno.models.talkinghead.FaceFormerDecoder import BertPriorDecoder 
from omegaconf import open_dict
from .VAEs import TVAE

def _create_squasher(self, type, input_dim, output_dim, quant_factor): 
    if type == "conv": 
        return ConvSquasher(input_dim, quant_factor, output_dim)
    elif type == "stack_linear": 
        return StackLinearSquash(input_dim, self.latent_frame_size, output_dim)
    else: 
        raise ValueError("Unknown squasher type")

class ConvSquasher(nn.Module): 

    def __init__(self, input_dim, quant_factor, output_dim) -> None:
        super().__init__()
        self.squasher = create_squasher(input_dim, output_dim, quant_factor)

    def forward(self, x):
        # BTF -> BFT 
        x = x.transpose(1, 2)
        x = self.squasher(x)
        # BFT -> BTF
        x = x.transpose(1, 2)
        return x

class StackLinearSquash(nn.Module): 
    def __init__(self, input_dim, latent_frame_size, output_dim): 
        super().__init__()
        self.input_dim = input_dim
        self.latent_frame_size = latent_frame_size
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim * latent_frame_size, output_dim)
        
    def forward(self, x):
        B, T, F = x.shape
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size
        F_stack = F * self.latent_frame_size
        x = x.reshape(B, T_latent, F_stack)
        x = x.view(B * T_latent, F_stack)
        x = self.linear(x)
        x = x.view(B, T_latent, -1)
        return x



class EMOTE(nn.Module) :
    def __init__(self, EMOTE_config, FLINT_config, FLINT_ckpt) :
        super(EMOTE, self).__init__()
        ## audio encoder
        self.audio_encoder = Wav2Vec2Encoder(EMOTE_config['audio_config']['model_specifier'], 
            EMOTE_config['audio_config']['trainable'], 
            with_processor=EMOTE_config['audio_config']['with_procesor'], 
            expected_fps=EMOTE_config['audio_config']['model_expected_fps'], # 50 fps is the default for wav2vec2 (but not sure if this holds universally)
            target_fps=EMOTE_config['audio_config']['target_fps'], # 25 fps is the default since we use 25 fps for the videos 
            freeze_feature_extractor=EMOTE_config['audio_config']['freeze_feature_extractor'])
        input_feature = self.audio_encoder.output_feature_dim()
        # sequence encoder
        decoder_config = EMOTE_config['sequence_decoder_config']
        self.audio_map = LinearSequenceEncoder(input_feature, decoder_config['feature_dim'])
        ## style encoder
        style_config = decoder_config['style_embedding']
        style_dim = style_config['n_intensities'] + style_config['n_identities'] + style_config['n_expression']
        self.style_encoder = nn.Linear(style_dim, decoder_config['feature_dim'])
        ## decoder
        # transformer encoder
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=decoder_config['feature_dim'] * dim_factor, 
                    nhead=decoder_config['nhead'], dim_feedforward=dim_factor*decoder_config['feature_dim'], 
                    activation=decoder_config['activation'],
                    dropout=decoder_config['dropout'], batch_first=True
        )        
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=decoder_config['num_layers'])
        # Squasher
        if decoder_config['squash_after'] :
            self.squasher = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], decoder_config['feature_dim'], decoder_config['quant_factor'])
        elif decoder_config['squash_before'] :
            self.squasher = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim']*dim_factor, decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'])
        else : 
            raise ValueError("Unknown squasher type")

        # Temporal VAE decoder
        self.decoder = TVAE(FLINT_config)
        decoder_ckpt = torch.load(FLINT_ckpt)
        self.decoder.load_state_dict(decoder_ckpt)
        # decoder freeze



        # deocoder_cfg = config['sequence_decoder']
        # with open_dict(decoder_cfg):
        #     decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
        #     decoder_cfg.predict_exp = cfg.model.output.predict_expcode
        #     decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        # decoder = BertPriorDecoder(decoder_cfg)

    def encode_audio(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        output = self.audio_encoder(audio) #(BS,768,T)
        output = self.audio_map(output) # (BS,128,T)
        # output = self.audio_encoder(audio).last_hidden_state # (BS,50*sec,768) : (BS,20,768)
        # output = self.audio_map(output) # (BS, T, 128)
        # output = output.transpose(1,2) # (BS, 128, T)
        # output = self.audio_feature_map(output) # (BS,50*sec,256) : (BS,20,512)
        return output # (BS, 128, T)

    def encode_style(self, condition) :
        '''
        condition : (BS, condition_num)
        condition_num = emotion + intensity + actors = 53
        '''
        output = self.style_encoder(condition) # (BS, condition_num)
        output = output.unsqueeze(2) # (BS, 128, 1)

        return output

    def decode(self, sample) :
        output = self.transformer_encoder(sample)
        output = self.squasher(output)
        output = self.decoder(sample)

        return output

    def forward(self, audio, condition) :
        audio_embedding = encode_audio(audio) # (BS, 128, T)
        style_embedding = encode_style(condition) # (BS, 128, 1)
        styled_audio_cat = torch.cat([hidden_states, style_emb], dim=-1)
        decode(styled_audio_cat)

        # audio_style_sum = torch.cat([audio_embedding, style_embedding], dim=2)
