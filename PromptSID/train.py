# flake8: noqa
import os.path as osp
import sys

sys.path.append('./')
sys.path.append('../')
from PromptSID.train_pipeline import train_pipeline

import PromptSID.archs
import PromptSID.data
import PromptSID.models
import PromptSID.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
