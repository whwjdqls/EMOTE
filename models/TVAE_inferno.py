import functools
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import VectorQuantizer
from .base_models import Transformer, PositionEmbedding,\
                                LinearEmbedding

from models.temporal import TransformerMasking, PositionalEncodings

# Following EMOTE paper,
# λrec is set to 1000000 and λKL to 0.001, which makes the
# converged KL divergence term less than one order of magnitude
# lower than the reconstruction terms
def calc_vae_loss(pred,target,mu, logvar, recon_weight=1000000, kl_weight=0.001):                            
    """ function that computes the various components of the VAE loss """
    reconstruction_loss = nn.MSELoss()(pred, target)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_weight * reconstruction_loss + kl_weight * KLD
                                      
def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """

    exp_loss = nn.L1Loss()(pred[:,:,:50], target[:,:,:50])
    rot_loss = nn.L1Loss()(pred[:,:,50:53], target[:,:,50:53])
    jaw_loss = alpha * nn.L1Loss()(pred[:,:,53:], target[:,:,53:])
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean() * quant_loss_weight + \
            (exp_loss + rot_loss + jaw_loss)
            

class TVAE(nn.Module):
    """Temporal VAE using Transformer backbone
    
    """
    def __init__(self, config):
        super().__init__()
        self.config = config['transformer_config']
        self.motion_encoder = TransformerEncoder(self.config)
        self.motion_decoder = TransformerDecoder(self.config)
        # self.mean = nn.Linear(self.config['hidden_size'], \
        #                             self.config['hidden_size'])
        # self.logvar = nn.Linear(self.config['hidden_size'], \
        #                             self.config['hidden_size'])

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar) # takes exponential function (log var -> var)
    #     eps = torch.randn_like(std) # random noise (JB : to device?)
    #     return eps.mul(std).add_(mu) 

    def forward(self, inputs):
        z ,mu, logvar = self.motion_encoder(inputs) # (BS, T/q, 128)
        # mu = self.mean(encoder_features) # (BS, T/q, 128)
        # logvar = self.logvar(encoder_features) # (BS, T/q, 128)
        # z = self.reparameterize(mu, logvar)
        pred_recon = self.motion_decoder(z)
        return pred_recon, mu, logvar

# class TransformerEncoder(nn.Module):
class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        size=self.config['in_dim'] # 50 + 3 = 53
        dim=self.config['hidden_size'] # d = 128
        layers = [nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim))]
        for _ in range(1, self.config['quant_factor']):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(0.2, True),
                        nn.BatchNorm1d(dim),
                        nn.MaxPool1d(2)
                        )]
        self.squasher = nn.Sequential(*layers)
        # following INFERNO, use nn.TransformerEncoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.config['hidden_size'],
            nhead=self.config['num_attention_heads'],
            dim_feedforward=self.config['intermediate_size'],
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder_transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config['num_hidden_layers']
        )  
        # define positional encoding. if false, None
        if self.config['pos_encoding'] == "learned":
            self.encoder_pos_encoding= PositionEmbedding( # for L2L use learnable PE but for FLINT use ALIBI PE
                self.config["quant_sequence_length"],
                self.config['hidden_size'])
        else:
            self.encoder_pos_encoding = None
            
        # Temperal bias
        if self.config['temporal_bias'] == "alibi_future":
            self.attention_mask = TransformerMasking.init_alibi_biased_mask_future(
                self.config['num_attention_heads'], 1200)
        else:
            self.attention_mask = None
            
        self.encoder_linear_embedding = torch.nn.Linear( 
            self.config['hidden_size'],
            self.config['hidden_size'])

        self.mean = nn.Linear(self.config['hidden_size'], \
                                    self.config['hidden_size'])
        self.logvar = nn.Linear(self.config['hidden_size'], \
                                    self.config['hidden_size'])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # takes exponential function (log var -> var)
        eps = torch.randn_like(std) # random noise (JB : to device?)
        return eps.mul(std).add_(mu) 

    def forward(self, inputs):
    ## downdample into path-wise length seq before passing into transformer
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # (BS, T/q, 128)->(BS , 128, T/q) -> (BS, T/q, 128)
        encoder_features = self.encoder_linear_embedding(inputs)

        if self.encoder_pos_encoding is not None:
            decoder_features = self.encoder_pos_encoding(encoder_features)

        # add attention mask bias (if any)
        mask = None
        B, T = encoder_features.shape[:2]
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=encoder_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
                
        encoder_features = self.encoder_transformer(encoder_features, mask=mask)

        mu = self.mean(encoder_features) # (BS, T/q, 128)
        logvar = self.logvar(encoder_features) # (BS, T/q, 128)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
  

class TransformerDecoder(nn.Module):
    def __init__(self, config, is_audio=False):
        super().__init__()
        self.config = config
        self.out_dim = config['in_dim'] # 50 + 3 = 53
        size=self.config['hidden_size']
        dim=self.config['hidden_size']
        self.expander = nn.ModuleList()
        self.expander.append(nn.Sequential(
        # https://github.com/NVIDIA/tacotron2/issues/182
        # After installing torch, please make sure to modify the site-packages/torch/nn/modules/conv.py 
        # file by commenting out the self.padding_mode != 'zeros' line to allow for replicated padding 
        # for ConvTranspose1d as shown in the above link -> 
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        padding_mode='zeros'), # set to zero for now, change latter
                                        #padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim)))
        num_layers = self.config['quant_factor']+2 \
            if is_audio else self.config['quant_factor'] # we never take audio into account for TVAE -> num_layer = 3
        seq_len = self.config["sequence_length"]*4 \
            if is_audio else self.config["sequence_length"] # we never take audio into account for TVAE -> seq_len = 32
        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(0.2, True),
                                nn.BatchNorm1d(dim),
                                ))
        # following INFERNO, use nn.TransformerEncoder
        decoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.config['hidden_size'],
            nhead=self.config['num_attention_heads'],
            dim_feedforward=self.config['intermediate_size'],
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder_transformer = torch.nn.TransformerEncoder(
            decoder_layer,
            num_layers=self.config['num_hidden_layers']
        )  
        
        # positional Encoding
        if self.config['pos_encoding'] == "learned":
            self.decoder_pos_encoding = PositionEmbedding(
                seq_len,
                self.config['hidden_size'])
        else: # if self.config['pos_encoding'] == false
            self.decoder_pos_encoding = None
            
        # Temperal bias
        if self.config['temporal_bias'] == "alibi_future":
            self.attention_mask = TransformerMasking.init_alibi_biased_mask_future(
                self.config['num_attention_heads'], 1200)
        else:
            self.attention_mask = None
            
        # linear embedding
        self.decoder_linear_embedding = torch.nn.Linear(
            self.config['hidden_size'],
            self.config['hidden_size'])

        # smooth layer
        self.cross_smooth_layer=\
            nn.Conv1d(self.config['hidden_size'],
                    self.out_dim, 5, padding=2)

        # post transformer linear layer
        self.post_transformer_linear = nn.Linear(self.config['hidden_size'],self.config['hidden_size'])

    def forward(self, inputs):
        ## upsample into original length seq before passing into transformer
        for i, module in enumerate(self.expander):
            inputs = module(inputs.permute(0,2,1)).permute(0,2,1)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=1)
        decoder_features = self.decoder_linear_embedding(inputs)
        
        if self.decoder_pos_encoding is not None:
            decoder_features = self.decoder_pos_encoding(decoder_features)

        # add attention mask bias (if any)
        mask = None
        B, T = decoder_features.shape[:2]
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=decoder_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
                
        decoder_features = self.decoder_transformer(decoder_features, mask=mask)
        post_transformer_linear_features = self.post_transformer_linear(decoder_features)
        pred_recon = self.cross_smooth_layer(
                                    post_transformer_linear_features.permute(0,2,1)).permute(0,2,1)
        return pred_recon

    def _forward(self, inputs):
        ## forward function used in EMOTE 
        ## FLINTs quant size is 8 which means the temperal length reduces to 1/8
        ## however, EMOTE's quant size is 4, we have to map EMOTEs latents to FLINTs latents in 
        ## the second layer of the FLINT's expander!
        for i, module in enumerate(self.expander):
            if i == 0 :
                continue
            inputs = module(inputs.permute(0,2,1)).permute(0,2,1)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=1)
        decoder_features = self.decoder_linear_embedding(inputs)
        
        if self.decoder_pos_encoding is not None:
            decoder_features = self.decoder_pos_encoding(decoder_features)

        # add attention mask bias (if any)
        mask = None
        B, T = decoder_features.shape[:2]
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=decoder_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
                
        decoder_features = self.decoder_transformer(decoder_features, mask=mask)
        pred_recon = self.cross_smooth_layer(
                                    decoder_features.permute(0,2,1)).permute(0,2,1)
        return pred_recon
