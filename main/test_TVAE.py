import argparse
import logging
import os, random
import sys
sys.path.append('../') # add parent directory to import modules

import json
import numpy as np
import torch

from datasets import dataset
from models import VAEs
from models.flame_models import flame
from utils.extra import seed_everything

def get_recon_from_path(config, TVAE, FLAME, device, path):
    params = np.load(path)
    exp_param = torch.tensor(params[:,:50], dtype=torch.float32).to(device)
    jaw_param = torch.tensor(params[:,50:53], dtype=torch.float32).to(device)
    inputs = torch.cat([exp_param, jaw_param], dim=-1)
    inputs = inputs.reshape(-1, 32, 53) # 
    params_pred, _, _ = TVAE(inputs)
    exp_param_pred = params_pred[:,:,:50].to(device)
    jaw_param_pred = params_pred[:,:,50:53].to(device)
    vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred, jaw_param_pred, device)
    vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param, jaw_param, device)
    
    
    
    

def main(args):
    """test TVAE model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # use cpu for now
    print('using device', device)
    
    seed_everything(42)
      
    # models
    print("Loading Models...")
    TVAE = VAEs.TVAE(config)
    FLAME = flame.FLAME(config)
    
    print("Loading Checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    TVAE.load_state_dict(checkpoint['TVAE'])
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    print(args)
    
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)

    main(args)
    