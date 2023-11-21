import torch
import json

from models.EMOTE import EMOTE

## initialize
with open('/home/jisoo6687/EMOTE/configs/EMOTE/EMOTE_V1.json') as EMOTE_config:
    EMOTE_config = json.load(EMOTE_config)
    FLINT_config_path = EMOTE_config['motionprior_config']
with open(FLINT_config_path) as flint_config :
    FLINT_config = json.load(flint_config)
#Choose number of epoch for checkpoint
if args.last_ckpt :
    checks = sorted(glob.glob(f'{args.save_dir}/*.pt'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    epoch = os.path.basename(checks[-1]).split('_')[-1].split('.')[0]
elif args.best_ckpt :
    epoch = 'best'
else :
    epoch = args.num_ckpt

FLINT_ckpt = f'{FLINT_config['model_path']}/TVAE_{epoch}.pth'

model = EMOTE(EMOTE_config, FLINT_config, FLINT_ckpt)
##

## inference
input_audio = torch.randn((3,40960))
input_style = torch.eye(43)[[2,15,29]]

results = model(input_audio, input_style)
print(results)
##