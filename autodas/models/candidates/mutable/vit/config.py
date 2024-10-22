#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/config.py
"""

import os

from iopath.common.file_io import PathManagerFactory
from yacs.config import CfgNode

pathmgr = PathManagerFactory.get()

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# -------------------------- Knowledge distillation options -------------------------- #
_C.DISTILLATION = CfgNode()

# Intermediate layers distillation options
_C.DISTILLATION.ENABLE_INTER = False
_C.DISTILLATION.INTER_TRANSFORM = 'linear'
_C.DISTILLATION.INTER_LOSS = 'l2'
_C.DISTILLATION.INTER_TEACHER_INDEX = []
_C.DISTILLATION.INTER_STUDENT_INDEX = []
_C.DISTILLATION.INTER_WEIGHT = 2.5

# ------------------------------- Common model options ------------------------------- #
_C.MODEL = CfgNode()

_C.MODEL.IMG_SIZE = 224
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.LOSS_FUN = 'cross_entropy'

# -------------------------------- Transformer options ------------------------------- #
_C.TRANSFORMER = CfgNode()

_C.TRANSFORMER.PATCH_SIZE = 16
_C.TRANSFORMER.HIDDEN_DIM = None
_C.TRANSFORMER.DEPTH = None
_C.TRANSFORMER.NUM_HEADS = None
_C.TRANSFORMER.MLP_RATIO = None

_C.TRANSFORMER.LN_EPS = 1e-6
_C.TRANSFORMER.DROP_RATE = 0.0
_C.TRANSFORMER.DROP_PATH_RATE = 0.1
_C.TRANSFORMER.ATTENTION_DROP_RATE = 0.0

_C.PIT = CfgNode()

_C.PIT.STRIDE = 8

_C.AUTOFORMER_SEARCH_SPACE = CfgNode()

_C.AUTOFORMER_SEARCH_SPACE.HIDDEN_DIM = [192, 216, 240]
_C.AUTOFORMER_SEARCH_SPACE.MLP_RATIO = [3.5, 4.0]
_C.AUTOFORMER_SEARCH_SPACE.DEPTH = [12, 13, 14]
_C.AUTOFORMER_SEARCH_SPACE.NUM_HEADS = [3, 4]

_C.AUTOFORMER = CfgNode()
_C.AUTOFORMER.HIDDEN_DIM = None
_C.AUTOFORMER.MLP_RATIO = None
_C.AUTOFORMER.DEPTH = None
_C.AUTOFORMER.NUM_HEADS = None

_C.PIT_SEARCH_SPACE = CfgNode()
_C.PIT_SEARCH_SPACE.MLP_RATIO = [2, 4, 6, 8]
_C.PIT_SEARCH_SPACE.NUM_HEADS = [2, 4, 8]
_C.PIT_SEARCH_SPACE.DEPTH = [[1, 2, 3], [4, 6, 8], [2, 4, 6]]
_C.PIT_SEARCH_SPACE.BASE_DIM = [16, 24, 32, 40]

_C.PIT = CfgNode()
_C.PIT.STRIDE = 8
_C.PIT.BASE_DIM = None
_C.PIT.MLP_RATIO = None
_C.PIT.DEPTH = None
_C.PIT.NUM_HEADS = None

# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with pathmgr.open(cfg_file, 'w') as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, 'r') as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)
