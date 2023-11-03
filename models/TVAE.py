import functools
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_models import Transformer, PositionEmbedding,\
                                LinearEmbedding

class TransformerEncoder(nn.Module):
  """ 
  Encoder class for VQ-VAE with Transformer backbone 
  inputs : FLAME Parameter Sequce (BS, T, 53)
  outputs : Downsampled Flame parameter sequence (BS, T/q, 128) where q = 8
  """

  def __init__(self, config):
    super().__init__()
    self.config = config
    size=self.config['transformer_config']['in_dim'] # 50 + 3 = 53
    dim=self.config['transformer_config']['hidden_size'] # d = 128
    layers = [nn.Sequential(
                   nn.Conv1d(size,dim,5,stride=2,padding=2,
                             padding_mode='replicate'),
                   nn.LeakyReLU(0.2, True),
                   nn.BatchNorm1d(dim))]
    for _ in range(1, config['transformer_config']['quant_factor']):
        layers += [nn.Sequential(
                       nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                 padding_mode='replicate'),
                       nn.LeakyReLU(0.2, True),
                       nn.BatchNorm1d(dim),
                       nn.MaxPool1d(2)
                       )]
    self.squasher = nn.Sequential(*layers)

    self.encoder_transformer = Transformer(
        in_size=self.config['transformer_config']['hidden_size'], # 128
        hidden_size=self.config['transformer_config']['hidden_size'], # 128
        num_hidden_layers=\
                self.config['transformer_config']['num_hidden_layers'], # 12 -> for flint unknown
        num_attention_heads=\
                self.config['transformer_config']['num_attention_heads'], # 8 -> for flint unknown
        intermediate_size=\
                self.config['transformer_config']['intermediate_size']) # 256 -> for flint unknown
    
    self.encoder_linear_embedding = LinearEmbedding( 
        self.config['transformer_config']['hidden_size'],
        self.config['transformer_config']['hidden_size'])
    
    self.encoder_pos_embedding = PositionEmbedding( # for L2L use learnable PE but for FLINT use ALIBI PE
        self.config["transformer_config"]["quant_sequence_length"],
        self.config['transformer_config']['hidden_size'])
    
  def forward(self, inputs):
    ## downdample into path-wise length seq before passing into transformer
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # (BS, T/q, 128)->(BS , 128, T/q) -> (BS, T/q, 128)
    encoder_features = self.encoder_linear_embedding(inputs)
    encoder_features = self.encoder_pos_embedding(encoder_features)
    encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
    return encoder_features
  

class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

  def __init__(self, config, out_dim, is_audio=False):
    super().__init__()
    self.config = config
    size=self.config['transformer_config']['hidden_size']
    dim=self.config['transformer_config']['hidden_size']
    self.expander = nn.ModuleList()
    self.expander.append(nn.Sequential(
                   nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                      output_padding=1,
                                      padding_mode='replicate'),
                   nn.LeakyReLU(0.2, True),
                   nn.BatchNorm1d(dim)))
    num_layers = config['transformer_config']['quant_factor']+2 \
        if is_audio else config['transformer_config']['quant_factor'] # we never take audio into account for TVAE -> num_layer = 3
    seq_len = config["transformer_config"]["sequence_length"]*4 \
        if is_audio else config["transformer_config"]["sequence_length"] # we never take audio into account for TVAE -> seq_len = 32
    for _ in range(1, num_layers):
        self.expander.append(nn.Sequential(
                             nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                       padding_mode='replicate'),
                             nn.LeakyReLU(0.2, True),
                             nn.BatchNorm1d(dim),
                             ))
    self.decoder_transformer = Transformer(
        in_size=self.config['transformer_config']['hidden_size'],
        hidden_size=self.config['transformer_config']['hidden_size'],
        num_hidden_layers=\
            self.config['transformer_config']['num_hidden_layers'],
        num_attention_heads=\
            self.config['transformer_config']['num_attention_heads'],
        intermediate_size=\
            self.config['transformer_config']['intermediate_size'])
    self.decoder_pos_embedding = PositionEmbedding(
        seq_len,
        self.config['transformer_config']['hidden_size'])
    self.decoder_linear_embedding = LinearEmbedding(
        self.config['transformer_config']['hidden_size'],
        self.config['transformer_config']['hidden_size'])
    self.cross_smooth_layer=\
        nn.Conv1d(config['transformer_config']['hidden_size'],
                  out_dim, 5, padding=2)

  def forward(self, inputs):
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    ## upsample into original length seq before passing into transformer
    for i, module in enumerate(self.expander):
        inputs = module(inputs.permute(0,2,1)).permute(0,2,1)
        if i > 0:
            inputs = inputs.repeat_interleave(2, dim=1)
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = self.decoder_pos_embedding(decoder_features)
    decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
    pred_recon = self.cross_smooth_layer(
                                decoder_features.permute(0,2,1)).permute(0,2,1)
    return pred_recon