# encoding utf-8
# 2020.11.19
import torch
import numpy as np
import random
from utils import helper


def seed_init(seed):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
