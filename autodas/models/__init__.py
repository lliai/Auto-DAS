from .candidates import *  # noqa: F401, F403
from .mobilenetv2 import mobile_half
from .resnet import (resnet8, resnet8x4, resnet14, resnet20, resnet32,
                     resnet32x4, resnet44, resnet56, resnet110)
from .resnetv2 import ResNet50
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
}


def build_model(model_name, **kwargs):
    if model_name not in model_dict:
        raise KeyError('Unknown model name: {}'.format(model_name))
    return model_dict[model_name](**kwargs)
