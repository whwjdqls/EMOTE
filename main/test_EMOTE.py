import argparse
import logging
import os, random
import sys
sys.path.append('../') # add parent directory to import modules

import json
import numpy as np
import torch

from datasets import dataset
from models import TVAE_inferno, EMOTE_inferno
from models.flame_models import flame
from utils.extra import seed_everything


def main(args, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device', device)

    # loading FLINT checkpoint 
    FLINT_config_path = config['motionprior_config']['config_path']
    
    with open(FLINT_config_path) as f :
        FLINT_config = json.load(f) 
        
    # FLINT trained model checkpoint path
    FLINT_ckpt = config['motionprior_config']['checkpoint_path']
    # models
    print("Loading Models...")
    TalkingHead = EMOTE_inferno.EMOTE(config, FLINT_config, FLINT_ckpt)
    # print talkignhead state dict
    for param in TalkingHead.state_dict():
        print(param, "\t", TalkingHead.state_dict()[param].size())

    inferno_checkpoint = torch.load(args.checkpoint)['state_dict']

    # load weights that only exists in Talking head model from the inferno_cehckpoint
    inferno_checkpoint = {k: v for k, v in inferno_checkpoint.items() if k in TalkingHead.state_dict()}
    print("Loading Inferno Weights...")
    TalkingHead.load_state_dict(inferno_checkpoint)

    print("DONE")
    # print inferno state dict
    # for param in inferno_checkpoint:
    #     print(param, "\t", inferno_checkpoint[param].size())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    print(args)
    
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)

    main(args, config)
    




    # main(args)
    