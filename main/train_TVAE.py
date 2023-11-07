
import sys
sys.path.append('../') # add parent directory to import modules
import json
from datasets import dataset
from models import VAEs


if __name__ == "__main__":
    config_path = "../configs/FLINT/FLINT_V1.json"
    config = json.load(open(config_path))
    dataset = dataset.FlameDataset(config)
    TVAE = VAEs.TVAE(config)
    