from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from models.wav2vec import Wav2Vec2Encoder
import torch.nn as nn
import json

class EMOTE(nn.Module) :
    def __init__(self, EMOTE_config, FLINT_config, FLINT_ckpt) :
        super(EMOTE, self).__init__()
        ## audio encoder
        self.audio_model = Wav2Vec2Encoder(EMOTE_config['audio_config']['model_specifier'], 
            EMOTE_config['audio_config']['trainable'], 
            with_processor=EMOTE_config['audio_config']['with_processor'], 
            expected_fps=EMOTE_config['audio_config']['model_expected_fps'], # 50 fps is the default for wav2vec2 (but not sure if this holds universally)
            target_fps=EMOTE_config['audio_config']['target_fps'], # 25 fps is the default since we use 25 fps for the videos 
            freeze_feature_extractor=EMOTE_config['audio_config']['freeze_feature_extractor'])
        # input_feature = self.audio_model.output_feature_dim() #768
        # # sequence encoder
        # decoder_config = EMOTE_config['sequence_decoder_config']
        # self.sequence_encoder = LinearSequenceEncoder(input_feature, decoder_config['feature_dim'])
        # self.sequence_decoder = BertPriorDecoder(decoder_config, FLINT_config, FLINT_ckpt)


with open('/home/jisoo6687/EMOTE/configs/EMOTE/EMOTE_V1.json') as EMOTE_config:
    EMOTE_config = json.load(EMOTE_config)
    # FLINT_config_path = EMOTE_config['motionprior_config']['config_path']
    FLINT_config_path = '/home/jisoo6687/EMOTE/configs/FLINT/FLINT_V1_MEAD.json'

with open(FLINT_config_path, 'r') as flint_config :
    FLINT_config = json.load(flint_config)

epoch = 99

FLINT_ckpt = f'{FLINT_config["model_path"]}/TVAE_{epoch}.pth'

model = EMOTE(EMOTE_config, FLINT_config, FLINT_ckpt)
print(model.audio_model)