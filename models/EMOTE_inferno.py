import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from .wav2vec import Wav2Vec2Encoder
import torch.nn.functional as F
from .TVAE_inferno import TVAE
from .temporal.TransformerMasking import init_faceformer_biased_mask_future
# import copy
# import einops
# from .sequence_encoder import LinearSequenceEncoder, ConvSquasher, StackLinearSquash
# from inferno.models.talkinghead.FaceFormerDecoder import BertPriorDecoder 
# from omegaconf import open_dict

# from VAEs import TVAE
def calculate_vertice_loss(pred, target):
     reconstruction_loss = nn.MSELoss()(pred, target)
     return reconstruction_loss
 
                    ##(linear stack, 128 *2,  128,   3,          4)
def _create_squasher(type, input_dim, output_dim, quant_factor, latent_frame_size =4): 
    if type == "conv": 
        return ConvSquasher(input_dim, quant_factor, output_dim)
    elif type == "stack_linear": 
        return StackLinearSquash(input_dim, latent_frame_size, output_dim)
    else: 
        raise ValueError("Unknown squasher type")

class SequenceEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def input_feature_dim(self):
        raise NotImplementedError("Subclasses must implement this method")

    def output_feature_dim(self):
        raise NotImplementedError("Subclasses must implement this method")

class LinearSequenceEncoder(SequenceEncoder): 

    def __init__(self, input_feature_dim, output_feature_dim):
        super().__init__()
        # self.cfg = cfg
        # input_feature_dim = self.cfg.get('input_feature_dim', None) or self.cfg.feature_dim 
        # output_feature_dim = self.cfg.feature_dim
        self.linear = torch.nn.Linear(input_feature_dim, output_feature_dim)

    def forward(self, sample):
        # feat = sample[input_key] 
        # B, T, D -> B * T, D 
        feat = sample.view(-1, sample.shape[-1]) # (BS*64,768)
        out = self.linear(feat) # (BS*64,128)
        # B * T, D -> B, T, D
        out = out.view(sample.shape[0], sample.shape[1], -1) # (BS,64,128)
        return out

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim

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

class StackLinearSquash(nn.Module): #( 128 *2, 4, 128)
    def __init__(self, input_dim, latent_frame_size, output_dim): 
        super().__init__()
        self.input_dim = input_dim # 128*2 
        self.latent_frame_size = latent_frame_size  # 4
        self.output_dim = output_dim # 128
        self.linear = nn.Linear(input_dim * latent_frame_size, output_dim)
        # print(f'input dim : {self.input_dim * latent_frame_size}')
        
    def forward(self, x):
        B, T, F = x.shape # (BS,64,256)
        print(f'T : {T}')
        print(f'latent_frame_size : {self.latent_frame_size}')
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size
        F_stack = F * self.latent_frame_size
        x = x.reshape(B, T_latent, F_stack) # (BS,16,1024)
        x = x.view(B * T_latent, F_stack)
        x = self.linear(x)
        x = x.view(B, T_latent, -1)
        return x

class LinearEmotionCondition(nn.Module):
    def __init__(self, condition_dim, output_dim):
        super().__init__()
        self.map = nn.Linear(condition_dim, output_dim)

    def forward(self, sample):
        return self.map(sample)



class EMOTE(nn.Module) :
    def __init__(self, EMOTE_config, FLINT_config, FLINT_ckpt) :
        super(EMOTE, self).__init__()
        ## audio encoder
        self.audio_model = Wav2Vec2Encoder(EMOTE_config['audio_config']['model_specifier'], 
            EMOTE_config['audio_config']['trainable'], 
            with_processor=EMOTE_config['audio_config']['with_processor'], 
            expected_fps=EMOTE_config['audio_config']['model_expected_fps'], # 50 fps is the default for wav2vec2 (but not sure if this holds universally)
            target_fps=EMOTE_config['audio_config']['target_fps'], # 25 fps is the default since we use 25 fps for the videos 
            freeze_feature_extractor=EMOTE_config['audio_config']['freeze_feature_extractor'])
        input_feature = self.audio_model.output_feature_dim() #768
        # sequence encoder
        decoder_config = EMOTE_config['sequence_decoder_config']
        self.sequence_encoder = LinearSequenceEncoder(input_feature, decoder_config['feature_dim'])
        self.sequence_decoder = BertPriorDecoder(decoder_config, FLINT_config, FLINT_ckpt)


    def encode_audio(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 40960)
        '''
        output = self.audio_model(audio) # (BS,64,768)
        output = self.sequence_encoder(output) # (BS,64,128)

        return output # (BS,64,128)

    def forward(self, audio, condition) :

        audio_embedding = self.encode_audio(audio) # (BS,64,128)
        # audio_embedding = self.LinearSequenceEncoder(audio_embedding)
        output = self.sequence_decoder(condition, audio_embedding) # (BS,128,53)

        return output # (BS,128,53)

class BertPriorDecoder(nn.Module):
    def __init__(self, decoder_config, FLINT_config, FLINT_ckpt):
        super(BertPriorDecoder, self).__init__()

        ## style encoder
        style_config = decoder_config['style_embedding']
        style_dim = style_config['n_intensities'] + style_config['n_identities'] + style_config['n_expression'] # 43
        
        # print(f'style dim : {style_dim}')
        self.obj_vector = LinearEmotionCondition(style_dim, decoder_config['feature_dim'])
        ## decoder
        #mask
        max_len = 1200
        self.biased_mask = init_faceformer_biased_mask_future(num_heads = decoder_config['nhead'], max_seq_len = max_len, period=decoder_config['period'])
        # transformer encoder
        dim_factor = 2
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=decoder_config['feature_dim'] * dim_factor, 
                    nhead=decoder_config['nhead'], dim_feedforward=dim_factor*decoder_config['feature_dim'], 
                    activation=decoder_config['activation'],
                    dropout=decoder_config['dropout'], batch_first=True
        )        
        self.bert_decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=decoder_config['num_layers'])
        # decoder.decoder
        self.decoder = nn.Linear(dim_factor*decoder_config['feature_dim'], decoder_config['feature_dim'])
        # Squasher
        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2 = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], decoder_config['feature_dim'], decoder_config['quant_factor'], decoder_config['latent_frame_size'])
        elif decoder_config['squash_before'] :
            self.squasher_1 = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size'])
        else : 
            raise ValueError("Unknown squasher type")

        # Temporal VAE decoder
        # 11-21
        # Load only decoder from TVAE
        self.motion_prior = TVAE(FLINT_config).motion_decoder
        decoder_ckpt = torch.load(FLINT_ckpt)
        if 'state_dict' in decoder_ckpt:
            decoder_ckpt = decoder_ckpt['state_dict']
        # new_decoder_ckpt = decoder_ckpt.copy()
        motion_decoder_state_dict = {
            key.replace('motion_decoder.', ''): value
            for key, value in decoder_ckpt.items()
            if key.startswith('motion_decoder.')
        }
        self.motion_prior.load_state_dict(motion_decoder_state_dict)
        
        # freeze decoder
        for param in self.motion_prior.parameters():
            param.requires_grad = False



    def encode_style(self, condition) :
        '''
        condition : (BS, condition_num)
        condition_num = emotion + intensity + actors = 43
        '''
        output = self.obj_vector(condition) # (BS, condition_num)
        output = output.unsqueeze(1) # (BS,1,128)
        return output

    def decode(self, sample) :
        
        mask = self.biased_mask[:, :sample.shape[1], :sample.shape[1]].clone().detach().to(device=sample.device)
        if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(sample.shape[0], 1, 1)
        
        output = self.bert_decoder(sample, mask=mask) # (BS,64,256)
        output = self.decoder(sample) # (BS,16,128)
        print(f'decoder output : {output.shape}')
        output = self.squasher_2(output) # (BS,16,128)
        print(f'squasher output : {output.shape}')
        # use the _forward function in the decoder which expands by quant factor 4
        # output = self.motion_prior.motion_decoder._forward(output) 
        output = self.motion_prior.forward(output) 
        # output = self.motion_prior._forward(output)
        return output

    def forward(self, condition, audio_embedding) :

        repeat_num = audio_embedding.shape[1]
        style_embedding = self.encode_style(condition).repeat(1,repeat_num,1) # (BS,64,128)

        styled_audio_cat = torch.cat([audio_embedding, style_embedding], dim=-1) # (BS,64,256)

        output = self.decode(styled_audio_cat) # (BS,128,53)

        return output # (BS,128,53)