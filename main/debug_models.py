import sys
sys.path.append('../') # add parent directory to import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datasets import dataset
from models import VAEs
from models.flame_models import flame



if __name__ =="__main__":
    BS = 1
    T = 32
    input = torch.randn(BS, T , 53)

    # config = json.load(open('C:\\Users\\jungbin.cho\\code\\EMOTE\\configs\\FLINT\\FLINT_V1.json'))
    config = json.load(open('../configs/FLINT/FLINT_V1.json'))

    model = VAEs.TVAE(config)
    output, mu, logvar = model(input)
    print(output.shape)
    print(mu.shape)
    print(logvar.shape)
    
    latents = torch.rand(BS, 8, 128)
    EMOTE_decoder = model.decoder
    
    out = EMOTE_decoder._forward(latents)
    print(out.shape)
    
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    # print(model.encoder.attention_mask)
    # print(model.encoder.attention_mask.shape)
    # Debug dataset
    # train_dataset = dataset.FlameDataset(config)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    # )
    
    # data = next(iter(train_dataloader))
    # print(data.shape)
    # exit()
    # print(exp_param.shape)
    # print(rot_pose.shape)
    # print(jaw_pose.shape)
    
    # flame_model = flame.FLAME(config)
    # vertices = flame.get_vertices_from_flame(config, flame_model, exp_param, jaw_pose)
    # print(vertices.shape)