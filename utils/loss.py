from inferno.utils.other import get_path_to_externals
from pathlib import Path
import sys
import torch
from inferno.models.temporal.Renderers import cut_mouth_vectorized
'''
E2E should be implemented from same version that LipReading used
'''
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
import argparse

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
        self.lip_reader = E2E(model_path, self.train_args)
        self.lip_reader.load_state_dict(torch.load(model_path))
        # self.lip_reader.to(devivce).eval()

        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        self.mouth_transform = Compose([
            Normalize(0.0, 1.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            Identity()]
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

    def crop_mouth(self, image, landmarks): 
        return cut_mouth_vectorized(image, 
                                    landmarks, 
                                    convert_grayscale=True, 
                                    mouth_window_margin = self.mouth_window_margin, 
                                    mouth_landmark_start_idx = self.mouth_landmark_start_idx, 
                                    mouth_landmark_stop_idx = self.mouth_landmark_stop_idx, 
                                    mouth_crop_height = self.mouth_crop_height, 
                                    mouth_crop_width = self.mouth_crop_width
                                    )