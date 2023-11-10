import argparse
import logging
import os, random
import sys
sys.path.append('../') # add parent directory to import modules

import json
import numpy as np
import torch
import wandb

from datasets import dataset
from models import VAEs
from models.flame_models import flame

def seed_everything(seed: int): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
# https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
# Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False causes 
# cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    torch.backends.cudnn.benchmark = False # -> Might want to set this to True if it's too slow
    


def train_one_epoch(config, epoch, model, FLAME, optimizer, data_loader, device):
    """
    Train the model for one epoch
    """
    model.train()
    model.to(device)
    FLAME.to(device)
    train_loss = 0
    total_steps = len(data_loader)
    for i, data in enumerate(data_loader):
        exp_param = data[:,:,:50].to(device)
        # rot_pose = data[:,:,50:53] # for MEAD dataset we have no rotation
        jaw_pose = data[:,:,50:53].to(device)
        
        inputs = torch.cat([exp_param, jaw_pose], dim=-1)
        
        params_pred, mu, logvar = model(inputs)
        exp_param_pred = params_pred[:,:,:50].to(device)
        # rot_pose = data[:,:,50:53]
        jaw_pose_pred = params_pred[:,:,50:53].to(device)

        vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred, jaw_pose_pred, device)
        vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param, jaw_pose, device)
        
        loss = VAEs.calc_vae_loss(vertices_pred, vertices_target, mu, logvar)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()
        if i % config["training"]["log_step"] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i * len(data),
                    len(data_loader.dataset),
                    100.0 * i / len(data_loader),
                    loss.item(),
                )
            )
        wandb.log({"train loss (step)": loss.detach().item()})
    avg_loss = train_loss / total_steps
    wandb.log({"train loss (epoch)": avg_loss})
    print("Train Epoch: {}\tAverage Loss: {:.6f}".format(epoch, avg_loss))
        
def val_one_epoch(config, epoch, model, FLAME, data_loader, device):
    model.eval()
    model.to(device)
    FLAME.to(device)
    val_loss = 0
    total_steps = len(data_loader)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            exp_param = data[:,:,:50].to(device)
            # rot_pose = data[:,:,50:53]
            jaw_pose = data[:,:,50:53].to(device)
            
            inputs = torch.cat([exp_param, jaw_pose], dim=-1)
            
            params_pred, mu, logvar = model(inputs)
            exp_param_pred = params_pred[:,:,:50].to(device)
            # rot_pose = data[:,:,50:53]
            jaw_pose_pred = params_pred[:,:,50:53].to(device)

            vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param, jaw_pose, device)
            
            loss = VAEs.calc_vae_loss(vertices_pred, vertices_target, mu, logvar)
            val_loss += loss.detach().item()
            if i % config["training"]["log_step"] == 0:
                print(
                    "Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        i * len(data),
                        len(data_loader.dataset),
                        100.0 * i / len(data_loader),
                        loss.item(),
                    )
                )
        avg_loss = val_loss / total_steps
        wandb.log({"val loss": avg_loss})
        print("Val Epoch: {}\tAverage Loss: {:.6f}".format(epoch, avg_loss))
        

def main(args):
    """training loop for TVAE (FLINT) in EMOTE
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # use cpu for now
    print('using device', device)
    
    seed_everything(42)
      
    # models
    print("Loading Models...")
    TVAE = VAEs.TVAE(config)
    FLAME = flame.FLAME(config)
    
    print("Loading Dataset...")
    # train_dataset = dataset.FlameDataset(config)
    train_dataset = dataset.MEADDataset(config, split='train')
    val_dataset = dataset.MEADDataset(config, split='val')
    print('val_dataset', len(val_dataset),'| train_dataset', len(train_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], drop_last=True)
    
    optimizer = torch.optim.Adam(TVAE.parameters(), lr=config["training"]["lr"])
    
    for epoch in range(0, config["training"]['num_epochs']):
        print('epoch', epoch, 'num_epochs', config["training"]['num_epochs'])
        train_one_epoch(config, epoch, TVAE, FLAME, optimizer, train_dataloader, device)
        val_one_epoch(config, epoch, TVAE, FLAME, val_dataloader, device)
        print("-"*50)

        if (epoch != 0) and (epoch % config["training"]["save_step"] == 0) :
            torch.save(
                TVAE.state_dict(),
                os.path.join(
                    config["training"]["save_dir"],
                    "TVAE_{}.pth".format(epoch),
                ),
            )
            print("Save model at {}\n".format(epoch))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    print(args)
    
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)
      
    wandb.init(project = config["project_name"],
            name = config["name"],
            config = config)
    
    main(args)
    
