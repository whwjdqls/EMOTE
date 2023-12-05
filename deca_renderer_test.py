import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from utils.renderer import SRenderY, set_rasterizer
import json
from utils import util
torch.backends.cudnn.benchmark = True

def test_render(model_cfg):
    set_rasterizer()
    uv_size = model_cfg['uv_size']
    device = torch.device('cuda:0')

    render = SRenderY(224, obj_filename=model_cfg['topology_path'], uv_size=uv_size, rasterizer_type='pytorch3d').to(device)
    # face mask for rendering details
    mask = imread(model_cfg['face_eye_mask_path']).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
    self.uv_face_eye_mask = F.interpolate(mask, [uv_size, uv_size]).to(device)
    mask = imread(model_cfg['face_mask_path']).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
    self.uv_face_mask = F.interpolate(mask, [uv_size,uv_size]).to(device)
    # displacement correction
    fixed_dis = np.load(model_cfg["fixed_displacement_path"])
    self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(device)
    # mean texture
    # mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
    # self.mean_texture = F.interpolate(mean_texture, [uv_size, uv_size]).to(device)
    # dense mesh template, for save detail mesh
    # self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()
    albedo = torch.zeros([1, 3, uv_size, uv_size], device=device) 
    ops = self.render(verts, trans_verts, albedo, h=224, w=224, background=background)
    ## output
    opdict['grid'] = ops['grid']
    opdict['rendered_images'] = ops['images']
    opdict['alpha_images'] = ops['alpha_images']
    opdict['normal_images'] = ops['normal_images']

if __name__ =='__main__':
    config_path = '/workspace/audio2mesh/EMOTE/configs/EMOTE/renderer.json'
    with open(config_path) as f:
        config = json.load(f)
    _setup_renderer(config)
