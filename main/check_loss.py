import json
import sys
sys.path.append('/home/jisoo6687/EMOTE')
from utils.loss import create_video_emotion_loss

with open('/home/jisoo6687/EMOTE/configs/EMOTE/EMOTE_inferno.json') as EMOTE_config:
    EMOTE_config = json.load(EMOTE_config)

LOSS_config = EMOTE_config['loss']['emotion_video_loss']
video_emotion_loss = create_video_emotion_loss(LOSS_config)


