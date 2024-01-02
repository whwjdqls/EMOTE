# from inferno.utils.other import get_path_to_externals
from pathlib import Path
import sys
import torch
import json
# from inferno.models.temporal.Renderers import cut_mouth_vectorized
'''
E2E should be implemented from same version that LipReading used
'''
sys.path.append('externals/spectre/external/Visual_Speech_Recognition_for_Multiple_Languages')
# from externals.spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
import argparse
import torchvision.transforms as t

## for video emotion loss
from omegaconf import OmegaConf
sys.path.append('models')
from video_emotion import SequenceClassificationEncoder, ClassificationHead, TransformerSequenceClassifier, MultiheadLinearClassificationHead, EmoCnnModule
import pytorch_lightning as pl 
from typing import Any, Optional, Dict, List
import torch.nn as nn


def check_nan(sample: Dict): 
    ok = True
    nans = []
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN found in '{key}'")
                nans.append(key)
                ok = False
                # raise ValueError("Nan found in sample")
    if len(nans) > 0:
        raise ValueError(f"NaN found in {nans}")
    return ok

# remove training part of VideoClassification
class VideoClassifierBase(pl.LightningModule): 

    def __init__(self, 
                 cfg, 
                #  preprocessor: Optional[Preprocessor] = None,
                #  feature_model: Optional[TemporalFeatureEncoder] = None,
                 preprocessor = None,
                 feature_model = None,
                 fusion_layer: Optional[nn.Module] = None,
                 sequence_encoder: Optional[SequenceClassificationEncoder] = None,
                 classification_head: Optional[ClassificationHead] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.feature_model = feature_model
        self.fusion_layer = fusion_layer
        self.sequence_encoder = sequence_encoder
        self.classification_head = classification_head

    def get_trainable_parameters(self):
        trainable_params = []
        if self.feature_model is not None:
            trainable_params += self.feature_model.get_trainable_parameters()
        if self.sequence_encoder is not None:
            trainable_params += self.sequence_encoder.get_trainable_parameters()
        if self.classification_head is not None:
            trainable_params += self.classification_head.get_trainable_parameters()
        return trainable_params

    @property
    def max_seq_length(self):
        return 5000

    @torch.no_grad()
    def preprocess_input(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        if self.preprocessor is not None:
            if self.device != self.preprocessor.device:
                self.preprocessor.to(self.device)
            sample = self.preprocessor(sample, input_key="video", train=train, test_time=not train, **kwargs)
        # sample = detach_dict(sample)
        return sample 

    def is_multi_modal(self):
        modality_list = self.cfg.model.get('modality_list', None) 
        return modality_list is not None and len(modality_list) > 1

    def forward(self, sample: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        """
        sample: Dict[str, torch.Tensor]
            - gt_emo_feature: (B, T, F)
        """
        # T = sample[input_key].shape[1]
        if "gt_emo_feature" in sample:
            T = sample['gt_emo_feature'].shape[1]
        else: 
            T = sample['video'].shape[1]
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            sample = truncate_sequence_batch(sample, self.max_seq_length)

        # preprocess input (for instance get 3D pseudo-GT )
        sample = self.preprocess_input(sample, train=train, **kwargs)
        check_nan(sample)

        if self.feature_model is not None:
            sample = self.feature_model(sample, train=train, **kwargs)
            check_nan(sample)
        else:
            input_key = "gt_emo_feature" # TODO: this needs to be redesigned 
            sample["hidden_feature"] = sample[input_key]


        # if self.is_multi_modal():
        #     sample = self.signal_fusion(sample, train=train, **kwargs)

        if self.sequence_encoder is not None:
            sample = self.sequence_encoder(sample) #, train=train, validation=validation, **kwargs)
            check_nan(sample)

        if self.classification_head is not None:
            sample = self.classification_head(sample)
            check_nan(sample)

        return sample

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoClassifierBase':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = VideoClassifierBase(cfg, prefix)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = VideoClassifierBase.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                strict=False, 
                **checkpoint_kwargs)
            # if stage == 'train':
            #     mode = True
            # else:
            #     mode = False
            # model.reconfigure(cfg, prefix, downgrade_ok=True, train=mode)
        return model
        
def truncate_sequence_batch(sample: Dict, max_seq_length: int) -> Dict:
    """
    Truncate the sequence to the given length. 
    """
    # T = sample["audio"].shape[1]
    # if max_seq_length < T: # truncate
    for key in sample.keys():
        if isinstance(sample[key], torch.Tensor): # if temporal element, truncate
            if sample[key].ndim >= 3:
                sample[key] = sample[key][:, :max_seq_length, ...]
        elif isinstance(sample[key], Dict): 
            sample[key] = truncate_sequence_batch(sample[key], max_seq_length)
        elif isinstance(sample[key], List):
            pass
        else: 
            raise ValueError(f"Invalid type '{type(sample[key])}' for key '{key}'")
    return sample

class VideoEmotionClassifier(VideoClassifierBase): 

    def __init__(self, 
                 cfg
        ):
        self.cfg = cfg
        preprocessor = None
        # feature_model = feature_enc_from_cfg(cfg.model.get('feature_extractor', None))
        feature_model = None
        fusion_layer = None
        if not self.is_multi_modal():
            feature_size = feature_model.output_feature_dim() if feature_model is not None else cfg.model.input_feature_size
        # else: 
        #     if self.cfg.model.fusion_type == 'tensor':
        #         assert len(self.cfg.model.modality_list) == 2 
        #         feature_size = ( cfg.model.input_feature_size + 1) * (feature_model.output_feature_dim() + 1) 
        #     elif self.cfg.model.fusion_type == 'tensor_low_rank': 
        #         assert len(self.cfg.model.modality_list) == 2 
        #         fusion_cfg = self.cfg.model.fusion_cfg
        #         fusion_layer = LowRankTensorFusion(fusion_cfg.rank, [cfg.model.input_feature_size, feature_model.output_feature_dim()], fusion_cfg.output_dim)
        #         feature_size = fusion_layer.output_feature_dim()
        #     else:
        #         feature_size = feature_model.output_feature_dim() + cfg.model.input_feature_size
        # sequence_classifier = sequence_encoder_from_cfg(cfg.model.get('sequence_encoder', None), feature_size)
        sequence_classifier = TransformerSequenceClassifier(cfg.model.get('sequence_encoder', None), feature_size)
        classification_head = MultiheadLinearClassificationHead(cfg.model.get('classification_head', None), 
                                                           sequence_classifier.encoder_output_dim(), 
                                                           cfg.model.output.num_classes,
                                                           )

        super().__init__(cfg,
            preprocessor = preprocessor,
            feature_model = feature_model,
            fusion_layer = fusion_layer,
            sequence_encoder = sequence_classifier,  
            classification_head = classification_head,  
        )


    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoEmotionClassifier':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = VideoEmotionClassifier(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = VideoEmotionClassifier.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **checkpoint_kwargs
            )
            # if stage == 'train':
            #     mode = True
            # else:
            #     mode = False
            # model.reconfigure(cfg, prefix, downgrade_ok=True, train=mode)
        return model

def locate_checkpoint(cfg_or_checkpoint_dir, replace_root = None, relative_to = None, mode=None, pattern=None):
    if isinstance(cfg_or_checkpoint_dir, (str, Path)):
        checkpoint_dir = str(cfg_or_checkpoint_dir)
    elif replace_root is not None:
        checkpoint_dir = str(Path(replace_root) / Path(cfg_or_checkpoint_dir.inout.checkpoint_dir))
    else :
        checkpoint_dir = cfg_or_checkpoint_dir.inout.checkpoint_dir

    if replace_root is not None and relative_to is not None:
        try:
            checkpoint_dir = str(Path(replace_root) / Path(checkpoint_dir).relative_to(relative_to))
        except ValueError as e:
            print(f"Not replacing the root of checkpoint_dir '{checkpoint_dir}' beacuse the specified root does not fit:"
                  f"'{replace_root}'")
    # if not Path(checkpoint_dir).is_absolute():
    #     checkpoint_dir = str(get_path_to_assets() / checkpoint_dir)
    print(f"Looking for checkpoint in '{checkpoint_dir}'")
    checkpoints = sorted(list(Path(checkpoint_dir).rglob("*.ckpt")))
    if len(checkpoints) == 0:
        print(f"Did not find checkpoints. Looking in subfolders")
        checkpoints = sorted(list(Path(checkpoint_dir).rglob("*.ckpt")))
        if len(checkpoints) == 0:
            print(f"Did not find checkpoints to resume from. Returning None")
            # sys.exit()
            return None
        print(f"Found {len(checkpoints)} checkpoints")
    else:
        print(f"Found {len(checkpoints)} checkpoints")
    if pattern is not None:
        checkpoints = [ckpt for ckpt in checkpoints if pattern in str(ckpt)]
    for ckpt in checkpoints:
        print(f" - {str(ckpt)}")

    if isinstance(mode, int):
        checkpoint = str(checkpoints[mode])
    elif mode == 'latest':
        # checkpoint = str(checkpoints[-1])
        checkpoint = checkpoints[0]
        # assert checkpoint.name == "last.ckpt", f"Checkpoint name is not 'last.ckpt' but '{checkpoint.name}'. Are you sure this is the right checkpoint?"
        if checkpoint.name != "last.ckpt":
            # print(f"Checkpoint name is not 'last.ckpt' but '{checkpoint.name}'. Are you sure this is the right checkpoint?")
            return None
        checkpoint = str(checkpoint)
    elif mode == 'best':
        min_value = 999999999999999.
        min_idx = -1
        # remove all checkpoints that do not containt the pattern 
        for idx, ckpt in enumerate(checkpoints):
            if ckpt.stem == "last": # disregard last
                continue
            end_idx = str(ckpt.stem).rfind('=') + 1
            loss_str = str(ckpt.stem)[end_idx:]
            try:
                loss_value = float(loss_str)
            except ValueError as e:
                print(f"Unable to convert '{loss_str}' to float. Skipping this checkpoint.")
                continue
            if loss_value <= min_value:
                min_value = loss_value
                min_idx = idx
        if min_idx == -1:
            raise FileNotFoundError("Finding the best checkpoint failed")
        checkpoint = str(checkpoints[min_idx])
    else:
        raise ValueError(f"Invalid checkpoint loading mode '{mode}'")
    print(f"Selecting checkpoint '{checkpoint}'")
    return checkpoint

def get_checkpoint_with_kwargs(cfg, prefix, replace_root = None, relative_to = None, checkpoint_mode=None, pattern=None):
    checkpoint = locate_checkpoint(cfg, replace_root = replace_root,
                                   relative_to = relative_to, mode=checkpoint_mode, pattern=pattern)
    cfg.model.resume_training = False  # make sure the training is not magically resumed by the old code
    # checkpoint_kwargs = {
    #     "model_params": cfg.model,
    #     "learning_params": cfg.learning,
    #     "inout_params": cfg.inout,
    #     "stage_name": prefix
    # }
    checkpoint_kwargs = {'config': cfg}
    return checkpoint, checkpoint_kwargs

def class_from_str(str, module=None, none_on_fail = False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")

def emo_network_from_path(path):
    print(f"Loading trained emotion network from: '{path}'")

    def load_configs(run_path):
        from omegaconf import OmegaConf
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        if run_path != conf.inout.full_run_dir: 
            conf.inout.output_dir = str(Path(run_path).parent)
            conf.inout.full_run_dir = str(run_path)
            conf.inout.checkpoint_dir = str(Path(run_path) / "checkpoints")
        return conf

    cfg = load_configs(path)

    if not bool(cfg.inout.checkpoint_dir):
        cfg.inout.checkpoint_dir = str(Path(path) / "checkpoints")

    checkpoint_mode = 'best'
    stages_prefixes = ""

    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, stages_prefixes,
                                                               checkpoint_mode=checkpoint_mode,
                                                               # relative_to=relative_to_path,
                                                               # replace_root=replace_root_path
                                                               )
    checkpoint_kwargs = checkpoint_kwargs or {}

    if 'emodeca_type' in cfg.model.keys():
        module_class = class_from_str(cfg.model.emodeca_type, sys.modules[__name__])
    else:
        module_class = EmoNetModule

    #module_class = EmoCnnModule
    emonet_module = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False,
                                                      **checkpoint_kwargs)
    return emonet_module

def metric_from_str(metric, **kwargs) :
    if metric == 'cosine_similarity' :
        return cosine_sim_negative

def cosine_sim_negative(*args, **kwargs) :
    return (1. - F.cosine_similarity(*args, **kwargs)).mean()


def create_video_emotion_loss(cfg): # cfg = EMOTE_config['loss']['emotion_video_loss']
    model_config_path = Path(cfg["network_path"]) / "cfg.yaml"
    # load config 
    model_config = OmegaConf.load(model_config_path)

    # sequence_model = load_video_emotion_recognition_net(cfg.network_path)
    class_ = class_from_str(model_config.model.pl_module_class, sys.modules[__name__])

    # instantiate the model
    checkpoint_mode = 'best' # resuming in the same stage, we want to pick up where we left of
    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(
        model_config, "", replace_root = 'models/VideoEmotionRecognition/models',
        checkpoint_mode=checkpoint_mode,
        pattern="val"
        )
    # sequence_model = class_.instantiate(model_config, None, None, checkpoint, checkpoint_kwargs)
    sequence_model = class_.instantiate(model_config, None, None, checkpoint, checkpoint_kwargs)

    ## see if the model has a feature extractor
    feat_extractor_cfg = model_config.model.get('feature_extractor', None)

    # video_emotion_loss_cfg.network_path = str(Path(video_network_folder) / video_emotion_loss_cfg.video_network_name)
    # video_emotion_loss = create_video_emotion_loss( video_emotion_loss_cfg).to(device)
     
    # if feat_extractor_cfg is None and hasattr(sequence_model, 'feature_extractor_path'):
    if (feat_extractor_cfg is None or feat_extractor_cfg["type"] is False) and cfg['feature_extractor_path']:
        # default to the affecnet trained resnet feature extractor
        feature_extractor_path = Path(cfg["feature_extractor_path"])
        feature_extractor = emo_network_from_path(str(feature_extractor_path))
    elif cfg["feature_extractor"] == "no":
        feature_extractor = None
    else: 
        # feature_extractor_path = feat_extractor_cfg.path
        feature_extractor = None
    
    metric = metric_from_str(cfg['metric'])
    loss = VideoEmotionRecognitionLoss(sequence_model, metric, feature_extractor)
    return loss

class VideoEmotionRecognitionLoss(torch.nn.Module):

    def __init__(self, video_emotion_recognition : VideoEmotionClassifier, metric, feature_extractor=None, ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.video_emotion_recognition = video_emotion_recognition
        self.metric = metric

    def forward(self, input, target):
        raise NotImplementedError()

    def _forward_input(self, 
        input_images=None, 
        input_emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        with torch.no_grad():
            return self.forward(input_images, input_emotion_features, mask, return_logits)

    def _forward_output(self, 
        output_images=None, 
        output_emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        return self.forward(output_images, output_emotion_features, mask, return_logits)

    def forward(self, 
        images=None, 
        emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        assert images is not None or emotion_features is not None, \
            "One and only one of input_images or input_emotion_features must be provided"
        if images is not None:
            B, T = images.shape[:2]
        else: 
            B, T = emotion_features.shape[:2]
        if emotion_features is None:
            feat_extractor_sample = {"image" : images.view(B*T, *images.shape[2:])}
            emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)
            # result_ = self.model.forward_old(images)
        if mask is not None:
            emotion_features = emotion_features * mask

        video_emorec_batch = {
            "gt_emo_feature": emotion_features,
        }
        video_emorec_batch = self.video_emotion_recognition(video_emorec_batch)

        emotion_feat = video_emorec_batch["pooled_sequence_feature"]
        
        if return_logits:
            if "predicted_logits" in video_emorec_batch:
                predicted_logits = video_emorec_batch["predicted_logits"]
                return emotion_feat, predicted_logits
            logit_list = {}
            if "predicted_logits_expression" in video_emorec_batch:
                logit_list["predicted_logits_expression"] = video_emorec_batch["predicted_logits_expression"]
            if "predicted_logits_intensity" in video_emorec_batch:
                logit_list["predicted_logits_intensity"] = video_emorec_batch["predicted_logits_intensity"]
            if "predicted_logits_identity" in video_emorec_batch:
                logit_list["predicted_logits_identity"] = video_emorec_batch["predicted_logits_identity"]
            return emotion_feat, logit_list

        return emotion_feat


    def compute_loss(
        self, 
        input_images=None, 
        input_emotion_features=None,
        output_images=None, 
        output_emotion_features=None,
        mask=None, 
        return_logits=False,
        ):
        # assert input_images is not None or input_emotion_features is not None, \
        #     "One and only one of input_images or input_emotion_features must be provided"
        # assert output_images is not None or output_emotion_features is not None, \
        #     "One and only one of output_images or output_emotion_features must be provided"
        # # assert mask is None, "Masked loss not implemented for video emotion recognition"
        # if input_images is not None:
        #     B, T = input_images.shape[:2]
        # else: 
        #     B, T = input_emotion_features.shape[:2]

        # if input_emotion_features is None:
        #     feat_extractor_sample = {"image" : input_images.view(B*T, *input_images.shape[2:])}
        #     input_emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)
        # if output_emotion_features is None:
        #     feat_extractor_sample = {"image" : output_images.view(B*T, *output_images.shape[2:])}
        #     output_emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)

        # if mask is not None:
        #     input_emotion_features = input_emotion_features * mask
        #     output_emotion_features = output_emotion_features * mask

        # video_emorec_batch_input = {
        #     "gt_emo_feature": input_emotion_features,
        # }
        # video_emorec_batch_input = self.video_emotion_recognition(video_emorec_batch_input)

        # video_emorec_batch_output = {
        #     "gt_emo_feature": output_emotion_features,
        # }
        # video_emorec_batch_output = self.video_emotion_recognition(video_emorec_batch_output)

        # input_emotion_feat = video_emorec_batch_input["pooled_sequence_feature"]
        # output_emotion_feat = video_emorec_batch_output["pooled_sequence_feature"]

        if return_logits:
            input_emotion_feat, in_logits = self._forward_input(input_images, input_emotion_features, mask, return_logits=return_logits)
            output_emotion_feat, out_logits = self._forward_output(output_images, output_emotion_features, mask, return_logits=return_logits)
            return self._compute_feature_loss(input_emotion_feat, output_emotion_feat), in_logits, out_logits

        input_emotion_feat = self._forward_input(input_images, input_emotion_features, mask)
        output_emotion_feat = self._forward_output(output_images, output_emotion_features, mask)
        return self._compute_feature_loss(input_emotion_feat, output_emotion_feat)


    def _compute_feature_loss(self, input_emotion_feat, output_emotion_feat):
        loss = self.metric(input_emotion_feat, output_emotion_feat)
        # for i in range(input_emotion_feat.shape[0]):
        #     print("In:\t", input_emotion_feat[i:i+1,:5]) 
        #     print("Out:\t", output_emotion_feat[i:i+1,:5]) 
        #     print(self.metric(input_emotion_feat[i:i+1], output_emotion_feat[i:i+1]))
        return loss
    


class LipReadingNet(torch.nn.Module):

    def __init__(self, loss_cfg, device): 
        super().__init__()
        # cfg_path = get_path_to_externals() / "spectre" / "configs" / "lipread_config.ini"
        # config = ConfigParser()
        # config.read(cfg_path)

        # model_path = str(get_path_to_externals() / "spectre" / config.get("model","model_path"))
        # model_conf = str(get_path_to_externals() / "spectre" / config.get("model","model_conf"))
        
        # config.set("model", "model_path", model_path)
        # config.set("model", "model_conf", model_conf)

        # self.lip_reader = Lipreading(
        #     config,
        #     device=device
        # )
        model_path = loss_cfg['lip_reading_loss']['E2E']['model']['model_path']
        model_conf = loss_cfg['lip_reading_loss']['E2E']['model']['model_conf']

        with open(model_conf, 'rb') as f:
            confs = json.load(f)
        if isinstance(confs, dict):
            args = confs
        else :
            idim, odim, args = confs
            self.odim = odim
        self.train_args = argparse.Namespace(**args)

        # define lip reading model
        # self.lip_reader = E2E(model_path, self.train_args)
        self.lip_reader = E2E(odim, self.train_args)
        self.lip_reader.load_state_dict(torch.load(model_path))
        # self.lip_reader.to(devivce).eval()

        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        # self.mouth_transform = t.Compose([
        #     t.Normalize(0.0, 1.0),
        #     t.CenterCrop(crop_size),
        #     t.Normalize(mean, std),
        #     t.Identity()]
        # )
        self.mouth_transform = t.Compose([
            t.Normalize(0.0, 1.0),
            t.CenterCrop(crop_size),
            t.Normalize(mean, std)]
        )


    def forward(self, lip_images):
        """
        :param lip_images: (batch_size, seq_len, 1, 88, 88) or (seq_len, 1, 88, 88))
        """
        # this is my - hopefully fixed version of the forward pass
        # In other words, in the lip reading repo code, the following happens:
        # inferno/external/spectre/external/Visual_Speech_Recognition_for_Multiple_Languages/espnet/nets/pytorch_backend/backbones/conv3d_extractor.py
        # line 95:
        # B, C, T, H, W = xs_pad.size() # evaluated to: torch.Size([B, 1, 70, 88, 88]) - so the temporal window is collapsed into the batch size
        ndim = lip_images.ndim
        B, T = lip_images.shape[:2]
        rest = lip_images.shape[2:]
        if ndim == 5: # batched 
            lip_images = lip_images.view(B * T, *rest)
        elif ndim == 4: # single
            pass
        else: 
            raise ValueError("Lip images should be of shape (batch_size, seq_len, 1, 88, 88) or (seq_len, 1, 88, 88)")

        channel_dim = 1
        lip_images = self.mouth_transform(lip_images.squeeze(channel_dim)).unsqueeze(channel_dim)

        
        if ndim == 5:
            lip_images = lip_images.view(B, T, *lip_images.shape[2:])
        elif ndim == 4: 
            lip_images = lip_images.unsqueeze(0)
            lip_images = lip_images.squeeze(2)

        # the image is now of shape (B, T, 88, 88), the missing channel dimension is unsqueezed in the lipread net code

        lip_features = self.lip_reader.model.encoder(
            lip_images,
            None,
            extract_resnet_feats=True
        )
        return lip_features

class LipReadingLoss(torch.nn.Module):
    '''
    Use LipReading Loss via LipReadingLoss(mout_gt, mout_pred)
    '''

    def __init__(self, loss_cfg, device, loss='cosine_similarity', 
                mouth_crop_width = 96,
                mouth_crop_height = 96,
                mouth_window_margin = 12,
                mouth_landmark_start_idx = 48,
                mouth_landmark_stop_idx = 68,
                ):
        super().__init__()
        self.loss = loss
        assert loss in ['cosine_similarity', 'l1_loss', 'mse_loss']
        self.model = LipReadingNet(loss_cfg, device)
        self.model.eval()
        # freeze model
        for param in self.parameters(): 
            param.requires_grad = False

        self.mouth_crop_width = mouth_crop_width
        self.mouth_crop_height = mouth_crop_height
        self.mouth_window_margin = mouth_window_margin
        self.mouth_landmark_start_idx = mouth_landmark_start_idx
        self.mouth_landmark_stop_idx = mouth_landmark_stop_idx

    def _forward_input(self, images):
        # there is no need to keep gradients for input (even if we're finetuning, which we don't, it's the output image we'd wannabe finetuning on)
        with torch.no_grad():
            result = self.model(images)
            # result_ = self.model.forward_old(images)
        return result

    def _forward_output(self, images):
        return self.model(images)
    
    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)

    def compute_loss(self, mouth_images_gt, mouth_images_pred, mask=None):
        lip_features_gt = self._forward_input(mouth_images_gt)
        lip_features_pred = self._forward_output(mouth_images_pred)

        lip_features_gt = lip_features_gt.view(-1, lip_features_gt.shape[-1])
        lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
        
        if mask is not None:
            lip_features_gt = lip_features_gt[mask.view(-1)]
            lip_features_pred = lip_features_pred[mask.view(-1)]
            # lip_features_gt = lip_features_gt[mask.squeeze(-1)]
            # lip_features_pred = lip_features_pred[mask.squeeze(-1)]
        
        return self._compute_feature_loss(lip_features_gt, lip_features_pred)
        # if self.loss == 'cosine_similarity':
        #     # pytorch cosine similarity
        #     lr = 1-torch.nn.functional.cosine_similarity(lip_features_gt, lip_features_pred, dim=1).mean()
        #     ## manual cosine similarity  take over from spectre
        #     # lr = (lip_features_gt*lip_features_pred).sum(1)/torch.linalg.norm(lip_features_pred,dim=1)/torch.linalg.norm(lip_features_gt,dim=1)
        #     # lr = 1 - lr.mean()
        # elif self.loss == 'l1_loss':
        #     lr = torch.nn.functional.l1_loss(lip_features_gt, lip_features_pred)
        # elif self.loss == 'mse_loss':
        #     lr = torch.nn.functional.mse_loss(lip_features_gt, lip_features_pred)
        # else:
        #     raise ValueError(f"Unknown loss function: {self.loss}")
        # return lr

    def _compute_feature_loss(self, lip_features_gt, lip_features_pred): 
        if self.loss == 'cosine_similarity':
            # pytorch cosine similarity
            lr = 1-torch.nn.functional.cosine_similarity(lip_features_gt, lip_features_pred, dim=1).mean()
            ## manual cosine similarity  take over from spectre
            # lr = (lip_features_gt*lip_features_pred).sum(1)/torch.linalg.norm(lip_features_pred,dim=1)/torch.linalg.norm(lip_features_gt,dim=1)
            # lr = 1 - lr.mean()
        elif self.loss == 'l1_loss':
            lr = torch.nn.functional.l1_loss(lip_features_gt, lip_features_pred)
        elif self.loss == 'mse_loss':
            lr = torch.nn.functional.mse_loss(lip_features_gt, lip_features_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        return lr

    # def crop_mouth(self, image, landmarks): 
    #     return cut_mouth_vectorized(image, 
    #                                 landmarks, 
    #                                 convert_grayscale=True, 
    #                                 mouth_window_margin = self.mouth_window_margin, 
    #                                 mouth_landmark_start_idx = self.mouth_landmark_start_idx, 
    #                                 mouth_landmark_stop_idx = self.mouth_landmark_stop_idx, 
    #                                 mouth_crop_height = self.mouth_crop_height, 
    #                                 mouth_crop_width = self.mouth_crop_width
    #                                 )