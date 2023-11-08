import argparse
import logging
import os
import sys
sys.path.append('../') # add parent directory to import modules
import json
import torch
from datasets import dataset
from models import VAEs
from models.flame_models import flame

def train_one_epoch(config, epoch, model, FLAME, optimizer, data_loader, device):
    """
    Train the model for one epoch
    """
    model.train()
    model.to(device)
    FLAME.to(device)
    for i, data in enumerate(data_loader):
        exp_param = data[:,:,:50].to(device)
        # rot_pose = data[:,:,50:53]
        jaw_pose = data[:,:,50:53].to(device)
        
        inputs = torch.cat([exp_param, jaw_pose], dim=-1)
        
        params_pred, mu, logvar = model(inputs)
        exp_param_pred = params_pred[:,:,:50].to(device)
        # rot_pose = data[:,:,50:53]
        jaw_pose_pred = params_pred[:,:,50:53].to(device)
        # print('exp_param_pred', exp_param_pred.shape)
        # print('jaw_pose_pred', jaw_pose_pred.shape)

        vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred, jaw_pose_pred, device)
        vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param, jaw_pose, device)
        
        loss = VAEs.calc_vae_loss(vertices_pred, vertices_target, mu, logvar)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
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
        
def test_one_epoch(config, epoch, model, FLAME, data_loader, device):
    pass

def main(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # use cpu for now
    print('using device', device)
    
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    print('using config', args.config)
    
    with open(args.config) as f:
      config = json.load(f)
      
    
    # models
    print("Loading Models...")
    TVAE = VAEs.TVAE(config)
    FLAME = flame.FLAME(config)
    
    print("Loading Dataset...")
    train_dataset = dataset.FlameDataset(config)
    train_dataset = dataset.MEADDataset(config, split='train')
    print('train_dataset', len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    
    optimizer = torch.optim.Adam(TVAE.parameters(), lr=config["training"]["lr"])
    
    for epoch in range(0, config["training"]['num_epochs']):
        print('epoch', epoch, 'num_epochs', config["training"]['num_epochs'])
        train_one_epoch(config, epoch, TVAE, FLAME, optimizer, train_dataloader, device)
        test_one_epoch(config, epoch, TVAE, FLAME, train_dataloader, device)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    print(args)
    args.config ="/home/whwjdqls99/EMOTE/configs/FLINT/FLINT_V1_MEAD.json" # manually set config path for now
    main(args)
    
