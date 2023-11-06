import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TVAE import TransformerEncoder,TransformerDecoder, TVAE
import json

class TempDownsampleConvBlock(nn.Module):
    '''
    input : FLAME Parameter Sequce (BS, 53, T)
    output : Downsampled Flame parameter sequence (BS, 53, T/q) where q = 8

    from 
    https://github.com/evonneng/learning2listen/blob/1f36508cba8fc8c0d41f12180aea274f234fb854/src/vqgan/vqmodules/gan_models.py#L178
    '''
    def __init__(self):
        super(TempDownsampleConvBlock, self).__init__()
        size = 53
        dim = 128 
        layers = [nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim))]
        for _ in range(1, 3):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(0.2, True),
                        nn.BatchNorm1d(dim),
                        nn.MaxPool1d(2)
                        )]
        self.squasher = nn.Sequential(*layers)

    def forward(self, x):
        x = self.squasher(x)
        return x


if __name__ =="__main__":
    BS = 1
    T = 32
    input = torch.randn(BS, T , 53)

    # config = json.load(open('C:\\Users\\jungbin.cho\\code\\EMOTE\\configs\\FLINT\\FLINT_V1.json'))
    config = json.load(open('../configs/FLINT/FLINT_V1.json'))
    model = TransformerEncoder(config)
    output = model(input)
    print(output.shape)
    
    model = TransformerDecoder(config, out_dim=53)
    output = model(output)
    print(output.shape)
    
    model = TVAE(config)
    output, mu, logvar = model(input)
    print(output.shape)
    print(mu.shape)
    print(logvar.shape)
