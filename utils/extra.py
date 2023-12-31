import os, random
import torch
import numpy as np

def seed_everything(seed: int): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    # Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False causes 
    # cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    torch.backends.cudnn.benchmark = False # -> Might want to set this to True if it's too slow

def class_from_str(str, module=None, none_on_fail = False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")

def get(cfg, key, value) :
    try :
        val = cfg[key]
    except :
        val = value
    return val

def get_checkpoint(cfg, replace_root = None, relative_to = None, checkpoint_mode=None, pattern=None):
    if checkpoint_mode is None:
        checkpoint_mode = 'latest'
        if hasattr(cfg['learning'], 'checkpoint_after_training'):
            checkpoint_mode = cfg['learning']['checkpoint_after_training']
    checkpoint = locate_checkpoint(cfg, replace_root = replace_root,
                                   relative_to = relative_to, mode=checkpoint_mode, pattern=pattern)
    return checkpoint

def get_checkpoint_with_kwargs(cfg, prefix, replace_root = None, relative_to = None, checkpoint_mode=None, pattern=None):
    checkpoint = get_checkpoint(cfg, replace_root = replace_root,
                                relative_to = relative_to, checkpoint_mode=checkpoint_mode, pattern=pattern)
    cfg['model']['resume_training'] = False  # make sure the training is not magically resumed by the old code
    # checkpoint_kwargs = {
    #     "model_params": cfg.model,
    #     "learning_params": cfg.learning,
    #     "inout_params": cfg.inout,
    #     "stage_name": prefix
    # }
    checkpoint_kwargs = {'config': cfg}
    return checkpoint, checkpoint_kwargs