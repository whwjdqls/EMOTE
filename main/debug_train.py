import glob
import json
import torch
import sys

sys.path.append('../') # add parent directory to import modules
from models import TVAE_inferno, EMOTE_inferno

with open('/workspace/audio2mesh/EMOTE/configs/EMOTE/EMOTE_v1_MEADv1_stage2.json') as f:
    EMOTE_config = json.load(f)

FLINT_config_path = EMOTE_config['motionprior_config']['config_path']
with open(FLINT_config_path) as f :
    FLINT_config = json.load(f) 
FLINT_ckpt = EMOTE_config['motionprior_config']['checkpoint_path']

TalkingHead = EMOTE_inferno.EMOTE(EMOTE_config, FLINT_config, FLINT_ckpt)
TalkingHead.load_state_dict(torch.load('/workspace/audio2mesh/EMOTE/checkpoints/EMOTE/v1_stage1_16/EMOTE_18.pth'))