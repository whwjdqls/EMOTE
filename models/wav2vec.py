import sys
import torch
import math
# from inferno.models.temporal.Bases import TemporalAudioEncoder
from torch.nn import Dropout
# from inferno.utils.other import class_from_str

from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, BaseModelOutput, Wav2Vec2BaseModelOutput
import torch.nn.functional as F
from .Bases import TemporalAudioEncoder


def temporal_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(math.ceil(seq_len * output_fps))
        # output_len = int(math.ceil(seq_len) * output_fps)
        # output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class Wav2Vec2ModelResampled(Wav2Vec2Model):

    """
    Wav2vec2 model with temporal resampling after the self.feature_extractor step. Everything else is identical to the base class.
    """

    def __init__(self, config, target_fps = 25, model_expected_fps = 50):
        super().__init__(config)
        self.model_expected_fps = model_expected_fps # default is 50 (at least for pretrained "facebook/wav2vec2-base-960h")
        self.target_fps = target_fps # default is 25 because that is what LRS3 dataset framerate is

    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        desired_output_length=None
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values) # (BS,512,49)
        extract_features = extract_features.transpose(1, 2) # (BS,49,512)

        ## This is where we do the temporal resampling, if necessary. It is the only difference to the base class.
        if self.model_expected_fps != self.target_fps or desired_output_length is not None:
            extract_features = temporal_interpolation(extract_features, self.model_expected_fps, self.target_fps, output_len=desired_output_length)
        ## End of temporal resampling. From here on, everything is identical to the base class.

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def _get_feat_extract_output_lengths(
        self, input_lengths, add_adapter = None
    ):
        input_lengths = super()._get_feat_extract_output_lengths(input_lengths, add_adapter)
        input_lengths = (input_lengths.to(torch.float32) / (self.model_expected_fps / self.target_fps)).to(torch.int64)

        return input_lengths


class Wav2Vec2ForSequenceClassificationResampled(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        # super().__init__(config)
        # call super of the parent of the base instead of the base class
        Wav2Vec2PreTrainedModel.__init__(self, config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = Wav2Vec2ModelResampled(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = torch.nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = torch.nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def _get_feat_extract_output_lengths(
        self, input_lengths, add_adapter
    ):
        input_lengths = super()._get_feat_extract_output_lengths(input_lengths, add_adapter)
        input_lengths = (input_lengths.to(torch.float32) / (self.wav2vec2.model_expected_fps / self.wav2vec2.target_fps)).to(torch.int64)

        return input_lengths

class Wav2Vec2Encoder(TemporalAudioEncoder):

    def __init__(self, model_specifier, trainable, with_processor=True, target_fps=25, expected_fps=50, 
                freeze_feature_extractor=True, 
                dropout_cfg=None,):
        super().__init__() 
        self.model_specifier = model_specifier
        self.cfg  =  Wav2Vec2Config.from_pretrained(model_specifier)
        # if dropout_cfg is not None:
        #     dropout_type = dropout_cfg.pop("type") 
        #     assert dropout_type is not None, "audio_dropout_cfg must have a 'type' key"
        #     self.dropout = class_from_str(dropout_type)(**dropout_cfg)
        # else: 
        #     self.dropout = None
        if with_processor:
            self.input_processor = Wav2Vec2Processor.from_pretrained(model_specifier)
        else: 
            self.input_processor = None
        # self.model = Wav2Vec2Model.from_pretrained(model_specifier)
        if not target_fps or not expected_fps:
            self.model = Wav2Vec2Model.from_pretrained(model_specifier)
            self.resampling = False
        else:
            self.model = Wav2Vec2ModelResampled.from_pretrained(model_specifier)
            self.resampling = True
            self.model.model_expected_fps = expected_fps
            self.model.target_fps = target_fps
        self.trainable = trainable
        if freeze_feature_extractor:
            self.model.feature_extractor._freeze_parameters()
        if not trainable: 
            self.model.requires_grad_(False)

    def get_trainable_parameters(self): 
        if self.trainable:
            return [p for p in self.model.parameters() if p.requires_grad]
        return []

    def _forward(self, sample, train=False, desired_output_length=None): 
    
        if self.input_processor is not None:
            B = sample["raw_audio"].shape[0]
            T = sample["raw_audio"].shape[1]
            # proc = self.input_processor(sample["raw_audio"], sampling_rate=sample["samplerate"], return_tensors="pt")[0]
            # raw_audio = sample["raw_audio"].view( B, -1)
            raw_audio = sample["raw_audio"].view( B, -1)
            proc = self.input_processor(raw_audio, sampling_rate=sample["samplerate"][0], return_tensors="pt")
            input = proc.input_values[0].to(device=raw_audio.device)
            sample["processed_audio"] = input
        else: 
            B = sample["processed_audio"].shape[0]
            # T = sample["processed_audio"].shape[1]
            T = None
            input = sample["processed_audio"].view( B, -1)
        if isinstance(self.model, Wav2Vec2ModelResampled):
            desired_output_length = desired_output_length or T
            feats_ = self.model(input, desired_output_length=desired_output_length)
            # feats_ = self.model(input)
        else:
            feats_ = self.model(input)
        F = feats_.last_hidden_state.shape[-1]
        T2 = feats_.last_hidden_state.shape[1]

        if self.resampling and T is not None:
            assert T2 == T # sanity checking that the feature got resampled to the proper length

        # sample["audio_feature"] = feats_.last_hidden_state 
        output = feats_.last_hidden_state

        if self.dropout is not None:
            # sample["audio_feature"] = self.dropout(sample["audio_feature"])
            output = self.dropout(output)

        # return sample
        return output

        # assert T2 + 1  == 2*T # Wav2Vec doubles the feature dimensionality and then this is reduced by 1 
        # # (probably because of a temporal convolution window of 3) 

        # # feats = torch.zeros((B, T2 + 1, F),
        # #     device=feats_.last_hidden_state.device, dtype=feats_.last_hidden_state.dtype)
        # # feats[:,:T2, :] = feats_.last_hidden_state
        
        # # feats = torch.zeros((B, T2, F),
        # #     device=feats_.last_hidden_state.device, dtype=feats_.last_hidden_state.dtype)
        # # padding = torch.zeros((B, 1, F),
        # #     device=feats_.last_hidden_state.device, dtype=feats_.last_hidden_state.dtype)

        # # feats = torch.cat((feats_.last_hidden_state, padding), dim=1).contiguous()

        # # TODO: question. The sequence length seems to have doubled. Why? Should I subsample the temporal dimension (i.e. interpolate) or should I reshape?
        # # 1) reshape
        # feats = feats.view(B, T, -1)
        # # 2) subsample - dummy version, only works when T2(+1) is a multiple of T
        # # feats = feats[:,::2,:]
        # # sample["audio_feature"] = feats 
        # # return sample

    def train(self, mode: bool = True):
        # return super().train(mode)
        mode = mode and self.trainable 
        self.model.train(mode)
        return self

    def forward(self, sample, train=False, desired_output_length=None): 
        if self.trainable:
            return self._forward(sample, train=train, desired_output_length=desired_output_length)
        else: 
            with torch.no_grad(): 
                return self._forward(sample, train=train, desired_output_length=desired_output_length)

    def output_feature_dim(self):
        return self.cfg.hidden_size
        # # return self.cfg.hidden_size * 2
        
        
# class Wav2Vec2SER(TemporalAudioEncoder):
#     """
#     Grokking notebook: https://colab.research.google.com/drive/1dNgohgrhtCqgZyqGJw91vOr4JEubZN_2
#     """
#     def __init__(self, model_specifier, trainable, with_processor=True, target_fps=25, expected_fps=50, 
#                 freeze_feature_extractor=False, 
#                 dropout_cfg=None,):
#         super().__init__() 
#         self.model_specifier = model_specifier
#         self.trainable = trainable
#         self.cfg = Wav2Vec2Config.from_pretrained(model_specifier)
#         if dropout_cfg is not None:
#             raise NotImplementedError("dropout not implemented yet")
#             # dropout_type = dropout_cfg.pop("type") 
#             # assert dropout_type is not None, "audio_dropout_cfg must have a 'type' key"
#             # self.dropout = class_from_str(dropout_type)(**dropout_cfg)
#         else: 
#             self.dropout = None
#         self.input_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_specifier)
#         if not target_fps or not expected_fps:
#             self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_specifier)
#             self.resampling = False
#         else:
#             # raise NotImplementedError("resampling not implemented yet")
#             self.model = Wav2Vec2ForSequenceClassificationResampled.from_pretrained(model_specifier)
#             self.resampling = True
#             self.model.wav2vec2.model_expected_fps = expected_fps
#             self.model.wav2vec2.target_fps = target_fps
#         if freeze_feature_extractor:
#             # self.model.feature_extractor._freeze_parameters()
#             self.model.freeze_feature_extractor()
#         if not trainable: 
#             self.model.requires_grad_(False)

#     def get_trainable_parameters(self): 
#         if self.trainable:
#             return [p for p in self.model.parameters() if p.requires_grad]
#         return []

#     def _forward(self, sample, train=False, desired_output_length=None): 
#         if self.input_processor is not None:
#             B = sample["raw_audio"].shape[0]
#             T = sample["raw_audio"].shape[1]
#             # proc = self.input_processor(sample["raw_audio"], sampling_rate=sample["samplerate"], return_tensors="pt")[0]
#             # raw_audio = sample["raw_audio"].view( B, -1)
#             raw_audio = sample["raw_audio"].view( B, -1)
#             proc = self.input_processor(raw_audio, sampling_rate=sample["samplerate"][0], padding=True, return_tensors="pt")
#             input = proc.input_values[0].to(device=raw_audio.device) # use index 0 because batch size is 1
#             sample["processed_audio"] = input
#         else: 
#             B = sample["processed_audio"].shape[0]
#             # T = sample["processed_audio"].shape[1]
#             T = None
#             input = sample["processed_audio"]
#         if isinstance(self.model, Wav2Vec2ForSequenceClassificationResampled):
#             # raise NotImplementedError("resampling not implemented yet")
            
#             out = self.model(input, return_dict=True, output_hidden_states=True)
#             logits = out.logits
#             # desired_output_length = desired_output_length or T
#             # feats_ = self.model(input, desired_output_length=desired_output_length)
#             # feats_ = self.model(input)
#         else:
#             out = self.model(input, output_dict=True, output_hidden_states=True)
#             logits = out.logits
            
#         predicted_ids = torch.argmax(logits, dim=-1)
#         labels = [self.model.config.id2label[_id] for _id in predicted_ids.tolist()]

        
#         sample["expression"] = F.softmax(logits) # shape (B, 4)
#         # sample["predicted_ids"] = predicted_ids # shape (B,)
#         sample["labels"] = labels # {0: 'neu', 1: 'hap', 2: 'ang', 3: 'sad'}, shape (B,)

#         sample["audio_emotion_feature"] = out.hidden_states[-1]

#         if self.dropout is not None:
#             raise NotImplementedError("dropout not implemented yet")
#             # sample["audio_emotion_feature"] = self.dropout(sample["audio_emotion_feature"])

#         return sample

#     def train(self, mode: bool = False):
#         # return super().train(mode)
#         # mode = mode and self.trainable 
#         # self.model.train(mode)
#         # return self
#         pass

#     def forward(self, sample, train=False, desired_output_length=None): 
#         if self.trainable:
#             raise NotImplementedError("SER is not trainable. Set trainable=False")
#             return self._forward(sample, train=train, desired_output_length=desired_output_length)
#         else: 
#             with torch.no_grad(): 
#                 return self._forward(sample, train=train, desired_output_length=desired_output_length)

#     def output_feature_dim(self):
#         return self.cfg.hidden_size
#         # # return self.cfg.hidden_size * 2