
import torch
import json
import sys
sys.path.append('../')
# from models.EMOTE import EMOTE

from utils.loss import LipReadingLoss

## initialize
with open('/workspace/audio2mesh/EMOTE/configs/EMOTE/EMOTE_V1.json') as EMOTE_config:
    EMOTE_config = json.load(EMOTE_config)
    # FLINT_config_path = EMOTE_config['motionprior_config']['config_path']
    FLINT_config_path = '/workspace/audio2mesh/EMOTE/configs/FLINT/FLINT_V1_MEAD.json'

# with open(FLINT_config_path, 'r') as flint_config :
#     FLINT_config = json.load(flint_config)

# lipreading model
device = 'cuda:0'
LOSS_config = EMOTE_config['loss']
dbg = LOSS_config['lip_reading_loss']['E2E']['model']
print(LOSS_config['lip_reading_loss']['E2E']['model']['model_path'])
# jisoo babo
# lip_reading_model = LipReadingLoss(device, LOSS_config, loss=LOSS_config['lip_reading_loss']['metric'])
# minsoo choigo
lip_reading_model = LipReadingLoss(LOSS_config, device, loss=LOSS_config['lip_reading_loss']['metric'])

lip_reading_model.to(device).eval()
lip_reading_model.requires_grad_(False)