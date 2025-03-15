# flake8: noqa
import os.path as osp
import sys

sys.path.append('./')
sys.path.append('../')
from basicsr.test import test_pipeline

import PromptSID.archs
import PromptSID.data
import PromptSID.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
