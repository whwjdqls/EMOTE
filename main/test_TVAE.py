import argparse
import logging
import os, random
import sys
sys.path.append('../') # add parent directory to import modules

import json
import numpy as np
import torch

from datasets import dataset
from models import TVAE_inferno
from models.flame_models import flame
from utils.extra import seed_everything

def get_recon(config, TVAE, FLAME, device, params, window_size=32):
    exp_param = torch.tensor(params[:,:50], dtype=torch.float32).to(device)
    jaw_param = torch.tensor(params[:,50:53], dtype=torch.float32).to(device)
    inputs = torch.cat([exp_param, jaw_param], dim=-1)
    recon_len = inputs.shape[0] - (inputs.shape[0] % window_size)
    inputs = inputs[:recon_len]
    inputs = inputs.reshape(-1, window_size, 53)
    TVAE.eval()
    with torch.no_grad():
        params_pred, _, _ = TVAE(inputs)
    exp_param_pred = params_pred[:,:,:50].to(device)
    jaw_param_pred = params_pred[:,:,50:53].to(device)
    # vertices_pred_batch = []
    # vertices_target_batch = []
    # for i in range(exp_param_pred.shape[0]):
    #     vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred[i].unsqueeze(0), jaw_param_pred[i].unsqueeze(0), device)
    #     vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param[i].unsqueeze(0), jaw_param[i].unsqueeze(0), device)
    #     vertices_pred_batch.append(vertices_pred)
    #     vertices_target_batch.append(vertices_target)
    # vertices_pred_batch = torch.stack(vertices_pred_batch, dim=0).squeeze(1)
    # vertices_target_batch = torch.stack(vertices_target_batch, dim=0).squeeze(1)
    return  inputs, params_pred

    
    
    
    

def main(args):
    """test TVAE model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # use cpu for now
    print('using device', device)
    
    seed_everything(42)
      
    # models
    print("Loading Models...")
    TVAE = TVAE_inferno.TVAE(config)
    FLAME = flame.FLAME(config, batch_size = 1 )
    FLAME.to(device)
    print("Loading Checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    inferno_checkpoint = torch.load('/workspace/audio2mesh/assets/MotionPrior/models/FLINT/checkpoints/model-epoch=0120-val/loss_total=0.131580308080.ckpt')["state_dict"]


    TVAE.load_state_dict(inferno_checkpoint)
    TVAE.to(device)


    param_path = '/mnt/storage/MEAD/flame_param/W016/W016_3_1_001.npy'
    params = np.load(param_path)
    inputs, params_pred = get_recon(config, TVAE, FLAME, device, params, 96)
    print(inputs.shape)
    print(inputs[0])
    print(params_pred.shape)
    print(params_pred[0])

    inputs = inputs.detach().cpu().numpy().reshape(-1, 53)
    params_pred = params_pred.detach().cpu().numpy().reshape(-1, 53)
    # np.save('W016_3_1_001_96_gt.npy', inputs)
    np.save('W016_3_1_001_96_inferno_pred.npy', params_pred)
    
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
    