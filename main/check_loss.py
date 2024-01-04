import json
import sys
sys.path.append('/home/jisoo6687/EMOTE')
from utils.loss import create_video_emotion_loss
import torch

with open('/home/jisoo6687/EMOTE/configs/EMOTE/EMOTE_inferno.json') as EMOTE_config:
    EMOTE_config = json.load(EMOTE_config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOSS_config = EMOTE_config['loss']['emotion_video_loss']
video_emotion_loss = create_video_emotion_loss(LOSS_config)
video_emotion_loss.to(device).eval()
video_emotion_loss.requires_grad_(False)

# pred_image = torch.randn(10,4,3,256,256).to("cuda:0") #(BS,T,256,256,3)
pred_image = torch.randn(10,4,3,224,224).to("cuda:0")
# gt_image = torch.randn(10,4,3,256,256).to("cuda:0")
gt_image = torch.randn(10,4,3,224,224).to("cuda:0")


# B = sample["predicted_vertices"].shape[0] # total batch size
B = pred_image.shape[0]
# T = sample["predicted_vertices"].shape[1] # sequence size
T = gt_image.shape[1]

# for disentangle
# B_orig = B // self.disentangle_expansion_factor(training, validation) # original batch size before disentanglement expansion
# effective batch size for computation of this particular loss 
# some loss functions should be computed on only the original part of the batch (vertex error)
# and some on everything (lip reading loss)

# for disentangle
# B_eff = B if loss_cfg.get('apply_on_disentangled', False) else B_orig
# B_eff = B if loss_cfg.get('apply_on_disentangled', True) else B_orig  ## WARNING, HERE TRUE BY DEFAULT
B_eff = B
# cam_name = list(sample["predicted_video"].keys())[0]
# assert len(list(sample["predicted_video"].keys())) == 1, "More cameras are not supported yet"
# rest = sample["predicted_video"][cam_name][:B_eff].shape[2:]
# rest = pred_image[:B_eff].shape[2:] #(H,W,C)
loss_values = {}
# for cam_name in sample["predicted_video"].keys():
# target_method = loss_cfg.get('target_method_image', None) # They use several methods for rendering, so needed but not for our case
# if target_method is None:
#     target_dict = sample
# else: 
#     target_dict = sample["reconstruction"][target_method]

# gt_vid = target_dict["gt_video"][cam_name][:B_eff]
# pred_vid = sample["predicted_video"][cam_name][:B_eff]
pred_vid = pred_image[:B_eff]
# loss_value = self.neural_losses.video_emotion_loss.compute_loss(
#     input_images=gt_vid, output_images=pred_vid,  mask=mask
#     )
# gt_emo_feature = sample["gt_emo_feature"][:B_eff]
gt_vid = gt_image[:B_eff]

# mask_ = mask[:B_eff, ...]
mask_ = None

loss_values = video_emotion_loss.compute_loss(input_images = gt_vid, output_images = pred_vid, mask=mask_)
# if "gt_emotion_video_features" in sample.keys():
#     gt_emo_feature = sample["gt_emotion_video_features"][cam_name][:B_eff]
#     predicted_emo_feature = self.neural_losses.video_emotion_loss._forward_output(pred_vid, mask=mask_)
#     # print("Emovideo loss:")
#     loss_values[cam_name] = self.neural_losses.video_emotion_loss._compute_feature_loss(gt_emo_feature, predicted_emo_feature)

# else:
#     loss_values[cam_name] = self.neural_losses.video_emotion_loss.compute_loss(
#         input_emotion_features=gt_emo_feature, output_images=pred_vid,  mask=mask_
#         )
print(f'loss values : {loss_values}')
