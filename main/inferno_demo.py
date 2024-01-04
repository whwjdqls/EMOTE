import torch
import json
import librosa
import numpy as np
from torch.utils.data._utils.collate import default_collate, default_collate_err_msg_format, np_str_obj_array_pattern, string_classes
import sys
import argparse

sys.path.append('../')
from models.EMOTE_inferno import EMOTE



def find_ckpt_epoch(args, type) :
    # replace this with actual ckpt path
    if args.last_ckpt :
        checks = sorted(glob.glob(f'{args.save_dir}/FLINT/*.pt'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        epoch = os.path.basename(checks[-1]).split('_')[-1].split('.')[0]
    elif args.best_ckpt :
        epoch = 'best'
    else :
        epoch = args.num_ckpt
    
    return epoch

def modify_EMOTE_ckpt(EMOTE_ckpt_path) :
    EMOTE_ckpt = torch.load(EMOTE_ckpt_path)['state_dict']
    new_EMOTE_ckpt = EMOTE_ckpt.copy()
    
    for key in EMOTE_ckpt.keys() :
        if key.startswith('renderer') or key.startswith('sequence_decoder.flame') :
            del new_EMOTE_ckpt[key]
        elif key.startswith('sequence_decoder.motion_prior.motion_decoder') :
            new_key = key.replace('sequence_decoder.motion_prior.motion_decoder', 'sequence_decoder.motion_prior')
            new_EMOTE_ckpt[new_key] = new_EMOTE_ckpt.pop(key)
    
    return new_EMOTE_ckpt

def read_audio(audio_path):
    sampling_rate = 16000
    # try:
    wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    # except ValueError: 
    #     import soundfile as sf
    #     wavdata, sampling_rate = sf.read(audio_path, channels=1, samplerate=16000,dtype=np.float32, subtype='PCM_32',format="RAW",endian='LITTLE')
    # wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    if wavdata.ndim > 1:
        wavdata = librosa.to_mono(wavdata)
    wavdata = (wavdata.astype(np.float64) * 32768.0).astype(np.int16)
    # if longer than 30s cut it
    if wavdata.shape[0] > 22 * sampling_rate:
        wavdata = wavdata[:22 * sampling_rate]
        print("Audio longer than 30s, cutting it to 30s")
    return wavdata, sampling_rate

def process_audio(wavdata, sampling_rate, video_fps):
    # assert sampling_rate % video_fps == 0 
    wav_per_frame = sampling_rate // video_fps 

    # As sampling_rate % video_fps != 0 , should add one dummy frame
    num_frames = wavdata.shape[0] // wav_per_frame +1

    wavdata_ = np.zeros((num_frames, wav_per_frame), dtype=wavdata.dtype) 
    wavdata_ = wavdata_.reshape(-1)
    if wavdata.size > wavdata_.size:
        wavdata_[...] = wavdata[:wavdata_.size]
    else: 
        wavdata_[:wavdata.size] = wavdata
    wavdata_ = wavdata_.reshape((num_frames, wav_per_frame))
    return wavdata_

def robust_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size. Copy of the default pytorch function, 
        but with some try-except blocks to better report errors when some samples are missing some fields.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return robust_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        print('FLOAT')
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        result = {}
        for key in elem: 
            try: 
                result[key] = robust_collate([d[key] for d in batch])
            except KeyError as e: 
                err = NestedKeyError(key)
                raise err 
            except NestedKeyError as e: 
                e.keys += [key]
                raise e
        return result
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(robust_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [robust_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([robust_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [robust_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
        
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def main(args, config) :
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device', device)
    
    # seed_everything(42)
    
    # loading FLINT checkpoint 
    FLINT_config_path = '/workspace/audio2mesh/EMOTE/configs/FLINT/FLINT_V1_MEADv1.json'
    with open(FLINT_config_path, 'r') as f :
        FLINT_config = json.load(f)
        
    FLINT_ckpt = '/workspace/audio2mesh/inferno/assets/MotionPrior/models/FLINT/checkpoints/model-epoch=0120-val/loss_total=0.131580308080.ckpt'
    
    # Load model
    model = EMOTE(config, FLINT_config, FLINT_ckpt)
    EMOTE_ckpt_path = '/workspace/audio2mesh/inferno/assets/TalkingHead/models/EMOTE/checkpoints/last.ckpt'
    EMOTE_ckpt = modify_EMOTE_ckpt(EMOTE_ckpt_path)
    model.load_state_dict(EMOTE_ckpt)
    
    wavdata, sampling_rate = read_audio(args.audio_path)
    sample = process_audio(wavdata, sampling_rate, 25)
    seq_length = sample.shape[0] - sample.shape[0] % 8
    # sample = sample[:seq_length,:]
    dl = torch.utils.data.DataLoader(TestDataset(sample), batch_size=seq_length, shuffle=False, num_workers=0, collate_fn=robust_collate)
    inputs = []
    outputs = []
    for bi, batch in enumerate(dl) :
        if batch.shape[0] !=seq_length :
            break
        else :
            batch = batch.view(-1)
            inputs.append(batch)

    input_dl = torch.utils.data.DataLoader(inputs, batch_size = args.batch_size, shuffle=False)
    # input_dl = torch.utils.data.DataLoader(TestDataset(sample), batch_size = args.batch_size, shuffle=False)
    input_style = torch.eye(43)[[3,9,31]]
    # input_style = torch.sum(input_style, dim=0).unsqueeze(0).repeat(len(input_dl),1)
    outputs = []
    for bi, batch in enumerate(input_dl) :
        input_style = torch.sum(input_style, dim=0).unsqueeze(0).repeat(batch.shape[0],1)
        batch = batch.type(torch.float32)
        output = model(batch, input_style)
        B,F,P = output.shape # Batch, Frame_num, Params
        output = output.reshape(B*F,P).detach().cpu().numpy()

        outputs.append(output)
    outputs = np.array(outputs).squeeze(0)
    print(f'flames shape : {outputs.shape}')
    np.save(f'{args.save_dir}.npy', outputs)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--EMOTE_config', type=str, default='/workspace/audio2mesh/EMOTE/configs/EMOTE/EMOTE_inferno.json')
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    
    with open(args.EMOTE_config) as f:
        EMOTE_config = json.load(f)
    
    main(args, EMOTE_config)
        