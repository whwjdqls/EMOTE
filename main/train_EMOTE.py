import argparse
import logging
import os, random
import sys
# sys.path.append('../') # add parent directory to import modules

import json
import numpy as np
import torch
import wandb

sys.path.append('.')
# from datasets import dataset
# from datasets import talkingheaddataset
from datasets import talkingheaddataset

from models import TVAE_inferno, EMOTE_inferno
from models.flame_models import flame
from utils.extra import seed_everything
from utils.loss import *
from utils.our_renderer import get_texture_from_template, render_flame, to_lip_reading_image, render_flame_lip

from torchvision import transforms
# from utils.loss import LipReadingLoss
import time
from tqdm import tqdm

def list_to(list_,device):
    """move a list of tensors to device
    """
    for i in range(len(list_)):
        list_[i] = list_[i].to(device)
    return list_

def label_to_condition_MEAD(config, emotion, intensity, actor_id):
    """labels to one hot condition vector
    """
    class_num_dict = config["sequence_decoder_config"]["style_embedding"]
    emotion = emotion - 1 # as labels start from 1
    emotion_one_hot = torch.nn.functional.one_hot(
        emotion, num_classes=class_num_dict["n_expression"])
    intensity = intensity - 1 # as labels start from 1
    intensity_one_hot = torch.nn.functional.one_hot(
        intensity, num_classes=class_num_dict["n_intensities"])
    # this might not be fair for validation set 
    actor_id_one_hot = torch.nn.functional.one_hot(
        actor_id, num_classes=class_num_dict["n_identities"]) # all actors
    condition = torch.cat([emotion_one_hot, intensity_one_hot, actor_id_one_hot], dim=-1) # (BS, 50)
    return condition.to(torch.float32)

def swap_conditions(original_batch) :
    batch_size, parameter_size, condition_size = original_batch.shape

    reshaped_batch = original_batch.view(batch_size, parameter_size, condition_size // 2, 2)

    swapped_batch = reshaped_batch.permute(0, 1, 3, 2).contiguous().view(batch_size, parameter_size, condition_size)

    return swapped_batch
    
    
    
def train_one_epoch(config,FLINT_config, epoch, model, FLAME, optimizer, data_loader, device):
    """
    Train the model for one epoch

    """
    model.train()
    train_loss = 0
    total_steps = len(data_loader)
    textures = None
    faces = None
    lip_reading_model = None
    # video_emotion_model = None

    epoch_start = time.time()
    for i, data_label in enumerate(tqdm(data_loader, desc="Processing", unit="step")) :
    # for i, data_label in enumerate(data_loader):
        forward_start = time.time()
        data, label = data_label # [data (audio, flame_param), label]
        audio, flame_param = list_to(data, device) # (BS, T / 30 * 16000), (BS, T, 53)
        BS, T = flame_param.shape[:2]

        emotion, intensity, gender, actor_id = list_to(label, device)
        condition = label_to_condition_MEAD(config, emotion, intensity, actor_id)
        params_pred = model(audio, condition) # batch, seq_len, 53

        # swap condition for disentanglement loss
        if epoch >= config["training"]["start_stage2"]:
            condition_size = condition.shape
            swapped_condition = condition.view(condition_size // 2, 2)
            swapped_condition = swapped_condition[:,:,[1,0]].view(condition_size)
            print(f'swapped_condition : {swapped_condition.shape}')
            swapped_params_pred = model(audio, swapped_condition)

            swapped_exp_param_pred = swapped_params_pred[:,:,50].to(device)
            swapped_jaw_pose_pred = swapped_params_pred[:,:,50].to(device)

            swapped_vertices_pred = flame.get_vertices_from_flame(
                FLINT_config, FLAME, swapped_exp_param_pred, swapped_jaw_pose_pred, device) # (BS, T, 15069)

        exp_param_pred = params_pred[:,:,:50].to(device)
        jaw_pose_pred = params_pred[:,:,50:53].to(device)
        exp_param_target = flame_param[:,:,:50].to(device)
        jaw_pose_target = flame_param[:,:,50:53].to(device)
        
        vertices_pred = flame.get_vertices_from_flame(
            FLINT_config, FLAME, exp_param_pred, jaw_pose_pred, device) # (BS, T, 15069)
        vertices_target = flame.get_vertices_from_flame(
            FLINT_config, FLAME, exp_param_target, jaw_pose_target, device) # (BS, T, 15069)

        recon_loss = calculate_vertice_loss(vertices_pred, vertices_target)
        
        loss = recon_loss.clone()
        lip_loss = torch.tensor(0.)
        if epoch >= config["training"]["start_stage2"]: # second stage (disentanglement / differential rendering)
            if textures is None: # load texture only once
                textures = get_texture_from_template( # this should be configged later
                    'models/flame_models/geometry/head_template.obj', device).extend(BS*T)
                faces = torch.tensor(FLAME.faces.astype(np.int64)).repeat(BS*T,1,1).to(device)

            #  12-06 is reshaping okay?
            vertices_target = vertices_target.reshape(BS*T, -1, 3) # (BS*T, 5023, 3)
            vertices_pred = vertices_pred.reshape(BS*T, -1, 3) # (BS*T, 5023, 3) 
            
            images_target = render_flame_lip(config, vertices_target, faces, textures, device)#(BS*T,88,88,4)
            images_pred = render_flame_lip(config, vertices_pred, faces, textures, device) # (BS*T,88,88,4)
           # images_target = render_flame(config, vertices_target, faces, textures, device) # (BS*T,256, 256,4)
           # images_pred = render_flame(config, vertices_pred, faces, textures, device) # (BS*T,256, 256,4)
            images_target = images_target[...,:3].permute(0,3,1,2)# (BS*T,3,88, 88) 
            images_pred = images_pred[...,:3].permute(0,3,1,2)# (BS*T,3,88, 88)

            lip_images_target = to_lip_reading_image(images_target)#(BS*T, 1, 1, 88, 88)
            lip_images_pred = to_lip_reading_image(images_pred)#(BS*T, 1, 1, 88, 88)

            if lip_reading_model is None:
                lip_reading_model = LipReadingLoss(config['loss'], device, loss=config['loss']['lip_reading_loss']['metric'])
                lip_reading_model.to(device).eval()
                lip_reading_model.requires_grad_(False)

            # input size of video emotion loss should be (BS,T,3,224,224)
            # if video_emotion_model is None :
            #     video_emotion_model = create_video_emotion(config['loss']['emotion_video_loss'])
            #     video_emotion_model.to(device).eval()
            #     video_emotion_model.requires_grad_(False)
 
            lip_loss = lip_reading_model(lip_images_target, lip_images_pred) * config['loss']['lip_reading_loss']['weight']
            loss += lip_loss
            # TODO 12-06 add disentanglement loss
            # loss += disentanglement loss
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()
        if i % config["training"]["log_step"] == 0:
            print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.10f}, recon loss:{:.10f}, lip loss:{:.10f}, time:{:.2f}".format(
                    epoch,
                    i * BS,
                    len(data_loader.dataset),
                    100.0 * i / len(data_loader),
                    loss.detach().item(),
                    recon_loss.detach().item(),
                    lip_loss.detach().item(),
                    time.time()-forward_start
                )
            )
        # DEBUGGGGG #
        # from torchvision.utils import save_image
        # print("saving images")
        # lip_images_target = lip_images_target.reshape(BS*T, 3, 88, 88) 
        # lip_images_pred = lip_images_pred.reshape(BS*T, 3, 88, 88)
        # for i in range(BS*T):
        #     image_tensor_pred = images_pred[i].detach().cpu()
        #     save_image(image_tensor_pred, f'/workspace/audio2mesh/EMOTE/results/test_training_code/whole/pred/{i + 1:03d}.png')
        #     image_tensor_target = images_target[i].detach().cpu()
        #     save_image(image_tensor_target, f'/workspace/audio2mesh/EMOTE/results/test_training_code/whole/gt/{i + 1:03d}.png')
        #     lip_image_tensor_pred = lip_images_pred[i].detach().cpu()
        #     save_image(lip_image_tensor_pred, f'/workspace/audio2mesh/EMOTE/results/test_training_code/lip/pred/{i + 1:03d}.png')
        #     lip_image_tensor_target = lip_images_target[i].detach().cpu()
        #     save_image(lip_image_tensor_target, f'/workspace/audio2mesh/EMOTE/results/test_training_code/lip/gt/{i + 1:03d}.png')
        ######################

        wandb.log({"train loss (step)": loss.detach().item()})
        wandb.log({"train recon loss (step)": recon_loss.detach().item()})
        wandb.log({"train lip loss (step)": lip_loss.detach().item()})

        if epoch >= config["training"]["start_stage2"]: 
            del images_target
            del images_pred
            del lip_images_target
            del lip_images_pred
            del vertices_target
            del vertices_pred
            del flame_param
            del audio
            del data
            torch.cuda.empty_cache()

    avg_loss = train_loss / total_steps
    wandb.log({"train loss (epoch)": avg_loss})
    print("Train Epoch: {}\tAverage Loss: {:.10f}, time: {:.2f}".format(epoch, avg_loss, time.time() - epoch_start))

def val_one_epoch(config,FLINT_config, epoch, model, FLAME, data_loader, device):
    """
    validage the model for one epoch
    """
    model.eval()
    val_loss = 0
    total_steps = len(data_loader)
    textures = None
    lip_reading_model = None
    faces = None
    with torch.no_grad():
        for i, data_label in enumerate(tqdm(data_loader, desc="Processing", unit="step")) :
        # for i, data_label in enumerate(data_loader):
            data, label = data_label
            # so many to(device) calls.. made a list_to function
            audio, flame_param = list_to(data, device)
            BS, T = flame_param.shape[:2]

            emotion, intensity, gender, actor_id = list_to(label, device)
            condition = label_to_condition_MEAD(config, emotion, intensity, actor_id)
            
            params_pred = model(audio, condition) 

            exp_param_pred = params_pred[:,:,:50].to(device)
            jaw_pose_pred = params_pred[:,:,50:53].to(device)
            exp_param_target = flame_param[:,:,:50].to(device)
            jaw_pose_target = flame_param[:,:,50:53].to(device)

            
            vertices_pred = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_target, jaw_pose_target, device)

            recon_loss = calculate_vertice_loss(vertices_pred, vertices_target)
            loss = recon_loss.clone()
            lip_loss = torch.tensor(0.)
            if epoch >= config["training"]["start_stage2"]: # second stage (disentanglement / differential rendering)
            # 12-10 Mabye we should load all the models in the main() function
                if textures is None: # load texture only once
                    textures = get_texture_from_template( # this should be configged later
                        'models/flame_models/geometry/head_template.obj', device).extend(BS*T)
                    faces = torch.tensor(FLAME.faces.astype(np.int64)).repeat(BS*T,1,1).to(device)

                #  12-06 is reshaping okay?
                vertices_target = vertices_target.reshape(BS*T, -1, 3) # (BS*T, 5023, 3)
                vertices_pred = vertices_pred.reshape(BS*T, -1, 3) # (BS*T, 5023, 3) 
                

                images_target = render_flame_lip(config, vertices_target, faces, textures, device)#(BS*T,88,88,4)
                images_pred = render_flame_lip(config, vertices_pred, faces, textures, device) # (BS*T,88,88,4)
                #images_target = render_flame(config, vertices_target, faces, textures, device)# (BS*T,256, 256,4)
                #images_pred = render_flame(config, vertices_pred, faces, textures, device) # (BS*T,256, 256,4)

                images_target = images_target[...,:3].permute(0,3,1,2)# (BS*T,3,256, 256)
                images_pred = images_pred[...,:3].permute(0,3,1,2)# (BS*T,3,256, 256)

                lip_images_target = to_lip_reading_image(images_target) #(BS*T, 1, 88, 88)
                lip_images_pred = to_lip_reading_image(images_pred) #(BS*T, 1, 88, 88)
       
                if lip_reading_model is None:
                    lip_reading_model = LipReadingLoss(config['loss'], device, loss=config['loss']['lip_reading_loss']['metric'])
                    lip_reading_model.to(device).eval()
                    lip_reading_model.requires_grad_(False)


                lip_loss = lip_reading_model(lip_images_target, lip_images_pred) * config['loss']['lip_reading_loss']['weight']
                loss += lip_loss
                # TODO 12-06 add disentanglement loss
                # loss += disentanglement loss
            val_loss += loss.detach().item()
            if i % config["training"]["log_step"] == 0:
                print(
                    "Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}, recon loss: {:.10f}, lip loss: {:.10f}".format(
                        epoch,
                        i * BS,
                        len(data_loader.dataset),
                        100.0 * i / len(data_loader),
                        loss.item(),
                        recon_loss.item(),
                        lip_loss.item()
                    )
                )
            if epoch >= config["training"]["start_stage2"]: 
                del images_target
                del images_pred
                del lip_images_target
                del lip_images_pred
                del vertices_target
                del vertices_pred
                del flame_param
                del audio
                del data
                torch.cuda.empty_cache()

        avg_loss = val_loss / total_steps
        wandb.log({"val loss": avg_loss})
        print("Val Epoch: {}\tAverage Loss: {:.10f}".format(epoch, avg_loss)) 

def test_one_epoch(config,FLINT_config, epoch, model, FLAME, data_loader, device):
    """
    SHOULD BE DEBUGGED FIRST
    test the model for one epoch
    this is for dataloaders with batch size 1
    """
    model.eval()
    val_loss = 0
    total_steps = len(data_loader)
    textures = None
    lip_reading_model = None
    with torch.no_grad():
        for i, data_label in enumerate(data_loader):
            data, label = data_label
            # so many to(device) calls.. made a list_to function
            audio, flame_param = list_to(data, device)
            # 11-21 JB BUGGGGGG
            # as we are using the while clip for validation set, 
            # sometimes the time dimension of the audio latent and 
            # exp param are different
            # for now we match this by cutting the longer one
            audio_len = audio.shape[1]
            seq_len = flame_param.shape[1]
            min_len = min(int(audio_len / 16_000 * 30), seq_len)
            # also, temporal len should be divisible by 4 due to EMOTEs quant factor
            min_len = min_len - (min_len % 8)
            # seq_len is 30 fps and audio sample rate is 16k
            # so, audio len should be seq_len / 30 * 16000
            # ex ) 180 seq_len (6 sec) -> 180 / 30 * 16000 = 96000 (6 sec)
            audio_len = int(min_len / 30 * 16_000)
            audio = audio[:,:audio_len]
            flame_param = flame_param[:,:min_len,:]
            BS, T = flame_param.shape[:2]

            emotion, intensity, gender, actor_id = list_to(label, device)
            # condition ([emotion, intensity, identity]])   
            condition = label_to_condition_MEAD(config, emotion, intensity, actor_id)
            
            params_pred = model(audio, condition) 

            exp_param_pred = params_pred[:,:,:50].to(device)
            jaw_pose_pred = params_pred[:,:,50:53].to(device)
            exp_param_target = flame_param[:,:,:50].to(device)
            jaw_pose_target = flame_param[:,:,50:53].to(device)

            
            vertices_pred = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_target, jaw_pose_target, device)
            print('vertices_pred', vertices_pred.shape, 'vertices_target', vertices_target.shape)
            
            recon_loss = calculate_vertice_loss(vertices_pred, vertices_target)
            loss = recon_loss
            lip_loss = torch.tensor(0.)
            # if epoch > 20: # second stage (disentanglement / differential rendering)
            # testing should be always stage 2
            # 12-10 Mabye we should load all the models in the main() function
            if textures is None: # load texture only once
                textures = get_texture_from_template( # this should be configged later
                    '/workspace/audio2mesh/EMOTE/models/flame_models/geometry/head_template.obj', device).extend(BS*T)
            faces = torch.tensor(FLAME.faces.astype(np.int64)).repeat(BS,1,1).to(device)

            #  12-06 is reshaping okay?
            vertices_target = vertices_target.reshape(BS*T, -1, 3) # (BS*T, 5023, 3)
            vertices_pred = vertices_pred.reshape(BS*T, -1, 3) # (BS*T, 5023, 3) 

            images_target = render_flame(config, vertices_target, faces, textures, device) # (BS*T, 3, 256, 256)
            images_pred = render_flame(config, vertices_pred, faces, textures, device) # (BS*T, 3, 256, 256)


            if lip_reading_model is None:
                lip_reading_model = LipReadingLoss(config['loss'], device, loss=config['loss']['lip_reading_loss']['metric'])
                lip_reading_model.to(device).eval()
                lip_reading_model.requires_grad_(False)
            lip_images_target = lip_images_target.reshape(BS*3,T, 1, 88, 88) # channel should be collapsed to batch
            lip_images_pred = lip_images_pred.reshape(BS*3,T, 1, 88, 88)

            lip_loss = lip_reading_model(lip_images_target, lip_images_pred) * config['loss']['lip_reading_loss']['weight']
            loss += lip_loss
            # TODO 12-06 add disentanglement loss
            # loss += disentanglement loss
     
            val_loss += loss.detach().item()
            if i % config["training"]["log_step"] == 0:
                print(
                    "Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}, recon loss: {:.10f}, lip loss: {:.10f}".format(
                        epoch,
                        i * BS,
                        len(data_loader.dataset),
                        100.0 * i / len(data_loader),
                        loss.item(),
                        recon_loss.item(),
                        lip_loss.item()
                    )
                )
        avg_loss = val_loss / total_steps
        wandb.log({"val loss": avg_loss})
        print("Val Epoch: {}\tAverage Loss: {:.6f}".format(epoch, avg_loss))
    
def main(args, config):
    """training loop for talkinghead model in EMOTE
    
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    TalkingHead = EMOTE_inferno.EMOTE(config, FLINT_config, FLINT_ckpt, load_motion_prior=False)
    if args.checkpoint is not None:
        print('loading checkpoint', args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        TalkingHead.load_state_dict(checkpoint)
        
    TalkingHead.to(device)
    # JB 11-21 have to have differnt flame models as it is initialized 
    # by batch size and train/val sets have different batch sizes
    # this can be improved by making FLAME invariant to batch size 
    # also, FLAME is currently initialized by EMOTE_config
    # I am not sure if this is the best way to do it
    FLAME_train = flame.FLAME(config, batch_size=config["training"]["batch_size"]).to(device).eval()
    FLAME_val = flame.FLAME(config, batch_size=config["validation"]["batch_size"]).to(device).eval()
    FLAME_train.requires_grad_(False)
    FLAME_val.requires_grad_(False)

    print("Loading Dataset...")
    # train_dataset = talkingheaddataset.TalkingHeadDataset_new(config, split='train')
    # train_dataset = talkingheaddataset.TalkingHeadDataset_new(config, split='debug')
    train_dataset = talkingheaddataset.TalkingHeadDataset_new(config, split='debug')

    # val_dataset = talkingheaddataset.TalkingHeadDataset_new(config, split='val')
    val_dataset = talkingheaddataset.TalkingHeadDataset_new(config, split='val')
    print('val_dataset', len(val_dataset),'| train_dataset', len(train_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["validation"]["batch_size"], drop_last=True)
    
    optimizer = torch.optim.Adam(TalkingHead.parameters(), lr=config["training"]["lr"])
    save_dir = os.path.join(config["training"]["save_dir"], config["name"])
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, config["training"]['num_epochs']+1):
        print('epoch', epoch, 'num_epochs', config["training"]['num_epochs'])

        training_time = time.time()
        train_one_epoch(config, FLINT_config, epoch, TalkingHead, FLAME_train, optimizer, train_dataloader, device)
        print('training time for this epoch :', time.time() - training_time)

        validation_time = time.time()
        val_one_epoch(config, FLINT_config, epoch, TalkingHead, FLAME_val, val_dataloader, device)
        print('validation time for this epoch :', time.time() - validation_time)
        print("-"*50)

        if epoch % config["training"]["save_step"] == 0 :
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
    parser.add_argument('--checkpoint', type=str, default=None, help = 'for stage2, we must give a checkpoint!')
    args = parser.parse_args()
    print(args)
    
    with open(args.EMOTE_config) as f:
        EMOTE_config = json.load(f)

        
    wandb.init(project = EMOTE_config["project_name"], # EMOTE
            name = EMOTE_config["name"], # test
            config = EMOTE_config) 
    
    main(args, EMOTE_config)
    
