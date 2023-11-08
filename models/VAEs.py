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


def calc_vae_loss(pred,target,mu, logvar  ):     
    # reduction_loss = F.binary_cross_entropy(pred, target, reduction='sum')
    reconstruction_loss = nn.MSELoss()(pred, target)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + KLD
                                           
def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """

    exp_loss = nn.L1Loss()(pred[:,:,:50], target[:,:,:50])
    rot_loss = nn.L1Loss()(pred[:,:,50:53], target[:,:,50:53])
    jaw_loss = alpha * nn.L1Loss()(pred[:,:,53:], target[:,:,53:])
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean() * quant_loss_weight + \
            (exp_loss + rot_loss + jaw_loss)
            
            
class VQVAE(nn.Module):
    """ VQ-VAE for motion prior learning 
    code adapted from https://github.com/evonneng/learning2listen
    """
    def __init__(self, config, version):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(
                                config, config['transformer_config']['in_dim'])
        self.quantize = VectorQuantizer(config['VQuantizer']['n_embed'],
                                        config['VQuantizer']['zquant_dim'],
                                        beta=0.25)
        
    def encode(self, x, x_a=None):
        h = self.encoder(x) ## x --> z'
        quant, emb_loss, info = self.quantize(h) ## finds nearest quantization
        return quant, emb_loss, info

    def decode(self, quant):
        dec = self.decoder(quant) ## z' --> x
        return dec

    def forward(self, x, x_a=None):
        quant, emb_loss, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, emb_loss

    def sample_step(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check

    def get_quant(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        indices = info[2]
        return quant_z, indices

    def get_distances(self, x):
        h = self.encoder(x) ## x --> z'
        d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0,2,1)
        x = self.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
        else:
            ix = logits
        ix = torch.reshape(ix, (-1,1))
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        """ function that samples the distribution of logits. (used in test)

        if sample_idx is None, we perform nucleus sampling
        """

        if sample_idx is None:
            ## nucleus sampling
            sample_idx = 0
            for b in range(logits.shape[0]):
                ## only take first prediction
                curr_logits = logits[b,0,:] / temperature
                assert curr_logits.dim() == 1
                sorted_logits, sorted_indices = torch.sort(curr_logits,
                                                           descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, \
                                                dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token
                # above the threshold
                sorted_indices_to_remove[..., 1:] = \
                                    sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                curr_logits[indices_to_remove] = filter_value
                logits[b,0,:] = curr_logits

        logits = logits[:,[0],:]
        probs = F.softmax(logits, dim=-1)
        if sample:
            ## multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1],shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx+1)[:,[-1]]
            probs = probs.reshape(shape[0],shape[1],shape[2])
            ix = ix.reshape(shape[0],shape[1],-1)
        else:
            ## top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs
    
class TVAE(nn.Module):
    """Temporal VAE using Transformer backbone
    
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config,config['transformer_config']['in_dim'])
        self.FC_mean = nn.Linear(config['transformer_config']['hidden_size'], \
                                    config['transformer_config']['hidden_size'])
        self.FC_logvar = nn.Linear(config['transformer_config']['hidden_size'], \
                                    config['transformer_config']['hidden_size'])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # takes exponential function (log var -> var)
        eps = torch.randn_like(std) # random noise (JB : to device?)
        return eps.mul(std).add_(mu) 

    def forward(self, inputs):
        encoder_features = self.encoder(inputs) # (BS, T/q, 128)
        mu = self.FC_mean(encoder_features) # (BS, T/q, 128)
        logvar = self.FC_logvar(encoder_features) # (BS, T/q, 128)
        z = self.reparameterize(mu, logvar)
        pred_recon = self.decoder(z)
        return pred_recon, mu, logvar
        
    
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