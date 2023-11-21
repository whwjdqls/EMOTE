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
from datasets import talkingheaddataset

from models import VAEs, EMOTE
from models.flame_models import flame
from utils.extra import seed_everything

def list_to(list_,device):
    """move a list of tensors to device
    """
    for i in range(len(list_)):
        list_[i] = list_[i].to(device)
    return list_

def label_to_condition_MEAD(emotion, intensity, actor_id):
    """labels to one hot condition vector
    """
    emotion = emotion - 1 # as labels start from 1
    emotion_one_hot = torch.nn.functional.one_hot(emotion, num_classes=9)
    intensity = intensity - 1 # as labels start from 1
    intensity_one_hot = torch.nn.functional.one_hot(intensity, num_classes=3)
    # this might not be fair for validation set 
    actor_id_one_hot = torch.nn.functional.one_hot(actor_id, num_classes=45) # all actors
    condition = torch.cat([emotion_one_hot, intensity_one_hot, actor_id_one_hot], dim=-1) # (BS, 50)
    return condition.to(torch.float32)
    
    
    
def train_one_epoch(config,FLINT_config, epoch, model, FLAME, optimizer, data_loader, device):
    """
    Train the model for one epoch
    """
    model.train()
    model.to(device)
    FLAME.to(device)
    train_loss = 0
    total_steps = len(data_loader)
    for i, data_label in enumerate(data_loader):
        data, label = data_label
        audio, flame_param = list_to(data, device)
        emotion, intensity, gender, actor_id = list_to(label, device)
        # condition ([emotion, intensity, identity]])   
        condition = label_to_condition_MEAD(emotion, intensity, actor_id)
        print('audio', audio.shape, 'condition', condition.shape)
            
        params_pred = model(audio, condition) # batch, seq_len, 53
        print("params_pred", params_pred.shape)
        exp_param_pred = params_pred[:,:,:50].to(device)
        jaw_pose_pred = params_pred[:,:,50:53].to(device)
        exp_param_target = flame_param[:,:,:50].to(device)
        jaw_pose_target = flame_param[:,:,50:53].to(device)

        vertices_pred = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_pred, jaw_pose_pred, device)
        vertices_target = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_target, jaw_pose_target, device)
        print('exp_param_pred', exp_param_pred.shape, 'jaw_pose_pred', jaw_pose_pred.shape)
        print('exp_param_target', exp_param_target.shape, 'jaw_pose_target', jaw_pose_target.shape)
        print('vertices_pred', vertices_pred.shape, 'vertices_target', vertices_target.shape)
            
        loss = EMOTE.calculate_vertice_loss(vertices_pred, vertices_target)
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
        # wandb.log({"train loss (step)": loss.detach().item()})
    avg_loss = train_loss / total_steps
    # wandb.log({"train loss (epoch)": avg_loss})
    print("Train Epoch: {}\tAverage Loss: {:.6f}".format(epoch, avg_loss))
    
def val_one_epoch(config,FLINT_config, epoch, model, FLAME, data_loader, device):
    model.eval()
    model.to(device)
    FLAME.to(device)
    val_loss = 0
    total_steps = len(data_loader)
    with torch.no_grad():
        for i, data_label in enumerate(data_loader):
            data, label = data_label
            # so many to(device) calls.. made a list_to function
            audio, flame_param = list_to(data, device)
            emotion, intensity, gender, actor_id = list_to(label, device)
            # condition ([emotion, intensity, identity]])   
            condition = label_to_condition_MEAD(emotion, intensity, actor_id)
            print('audio', audio.shape, 'condition', condition.shape)
            
            params_pred = model(audio, condition) 
            print('params_pred', params_pred.shape)
            
            exp_param_pred = params_pred[:,:,:50].to(device)
            jaw_pose_pred = params_pred[:,:,50:53].to(device)
            exp_param_target = flame_param[:,:,:50].to(device)
            jaw_pose_target = flame_param[:,:,50:53].to(device)
            print('exp_param_pred', exp_param_pred.shape, 'jaw_pose_pred', jaw_pose_pred.shape)
            print('exp_param_target', exp_param_target.shape, 'jaw_pose_target', jaw_pose_target.shape)
            
            vertices_pred = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_target, jaw_pose_target, device)
            print('vertices_pred', vertices_pred.shape, 'vertices_target', vertices_target.shape)
            
            loss = EMOTE.calculate_vertice_loss(vertices_pred, vertices_target)
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
        # wandb.log({"val loss": avg_loss})
        print("Val Epoch: {}\tAverage Loss: {:.6f}".format(epoch, avg_loss))
        
    
def main(args, config):
    """training loop for talkinghead model in EMOTE
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # use cpu for now
    print('using device', device)
    
    seed_everything(42)
    # loading FLINT checkpoint 
    FLINT_config_path = config['motionprior_config']['config_path']
    
    with open(FLINT_config_path) as f :
        FLINT_config = json.load(f) 
        
    # FLINT trained model checkpoint path
    FLINT_ckpt = config['motionprior_config']['checkpoint_path']
    # models
    print("Loading Models...")
    TalkingHead = EMOTE.EMOTE(config, FLINT_config, FLINT_ckpt)
    # JB 11-21 have to have differnt flame models as it is initialized 
    # by batch size and train/val sets have different batch sizes
    # this can be improved by making FLAME invariant to batch size 
    # also, FLAME is currently initialized by EMOTE_config
    # I am not sure if this is the best way to do it
    FLAME_train = flame.FLAME(config, split='train')
    FLAME_val = flame.FLAME(config, split='val')

    
    print("talkingHead state dict and shapes")
    for name, param in TalkingHead.named_parameters():
        print(name, param.shape)
        
    print("Loading Dataset...")
    # train_dataset = dataset.FlameDataset(config)
    train_dataset = talkingheaddataset.TalkingHeadDataset(config, split='train')
    data, labels = train_dataset[0]

    val_dataset = talkingheaddataset.TalkingHeadDataset(config, split='val')
    print('val_dataset', len(val_dataset),'| train_dataset', len(train_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["validation"]["batch_size"], drop_last=True)
    
    optimizer = torch.optim.Adam(TalkingHead.parameters(), lr=config["training"]["lr"])
    save_dir = os.path.join(config["training"]["save_dir"], config["name"])
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(0, config["training"]['num_epochs']):
        print('epoch', epoch, 'num_epochs', config["training"]['num_epochs'])
        train_one_epoch(config, FLINT_config, epoch, TalkingHead, FLAME_train, optimizer, train_dataloader, device)
        val_one_epoch(config, FLINT_config, epoch, TalkingHead, FLAME_val, val_dataloader, device)
        print("-"*50)

        if (epoch != 0) and (epoch % config["training"]["save_step"] == 0) :
            torch.save(
                TalkingHead.state_dict(),
                os.path.join(
                    config["training"]["save_dir"],
                    config["name"],
                    "EMOTE_{}.pth".format(epoch),
                ),
            )
            print("Save model at {}\n".format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--EMOTE_config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    print(args)
    
    with open(args.EMOTE_config) as f:
        EMOTE_config = json.load(f)

        
    # wandb.init(project = EMOTE_config["project_name"], # EMOTE
    #         name = EMOTE_config["name"], # test
    #         config = EMOTE_config) 
    
    main(args, EMOTE_config)
    