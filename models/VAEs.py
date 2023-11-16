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
            
            
class VQVAE(nn.Module):
    """ VQ-VAE for motion prior learning 
    code adapted from https://github.com/evonneng/learning2listen
    """
    def __init__(self, config, version):
        super().__init__()
        self.encoder = TransformerEncoder(config['transformer_config'])
        self.decoder = TransformerDecoder(config['transformer_config'])
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
        self.config = config['transformer_config']
        self.encoder = TransformerEncoder(self.config)
        self.decoder = TransformerDecoder(self.config)
        self.mean = nn.Linear(self.config['hidden_size'], \
                                    self.config['hidden_size'])
        self.logvar = nn.Linear(self.config['hidden_size'], \
                                    self.config['hidden_size'])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # takes exponential function (log var -> var)
        eps = torch.randn_like(std) # random noise (JB : to device?)
        return eps.mul(std).add_(mu) 

    def forward(self, inputs):
        encoder_features = self.encoder(inputs) # (BS, T/q, 128)
        mu = self.mean(encoder_features) # (BS, T/q, 128)
        logvar = self.logvar(encoder_features) # (BS, T/q, 128)
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
        
    self.encoder_linear_embedding = LinearEmbedding( 
        self.config['hidden_size'],
        self.config['hidden_size'])

        
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
    return encoder_features
  

class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

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
    self.decoder_linear_embedding = LinearEmbedding(
        self.config['hidden_size'],
        self.config['hidden_size'])

    # smooth layer
    self.cross_smooth_layer=\
        nn.Conv1d(self.config['hidden_size'],
                  self.out_dim, 5, padding=2)

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
    pred_recon = self.cross_smooth_layer(
                                decoder_features.permute(0,2,1)).permute(0,2,1)
    return pred_recon