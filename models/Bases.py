import torch
from typing import Dict, List, Optional, Tuple, Union, Any

class SequenceEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def input_feature_dim(self):
        raise NotImplementedError("Subclasses must implement this method")

    def output_feature_dim(self):
        raise NotImplementedError("Subclasses must implement this method")

class TemporalAudioEncoder(torch.nn.Module): 

    def __init__(self):
        super().__init__() 

    def forward(self, sample, train=False, desired_output_length=None, **kwargs): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()

    def output_feature_dim(self): 
        raise NotImplementedError()

def get(cfg, key, value) :
    try :
        val = cfg[key]
    except :
        val = value
    return val

class Preprocessor(object):

    def __init__(self):
        super().__init__() 
        
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def to(self, device):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def device(self):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def test_time(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_flametex(self):
        raise NotImplementedError("Subclasses must implement this method")