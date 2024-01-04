# from inferno.utils.other import get_path_to_externals
from pathlib import Path
import sys
import torch
import json
# from inferno.models.temporal.Renderers import cut_mouth_vectorized
'''
E2E should be implemented from same version that LipReading used
'''
sys.path.append('../externals/spectre/external/Visual_Speech_Recognition_for_Multiple_Languages')
# from externals.spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
import argparse
import torchvision.transforms as t
import math
import numpy as np
import torch
import torch.nn as nn
import pickle