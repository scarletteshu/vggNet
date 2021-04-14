import sys
sys.path.append('../')

from models.vggNet import *

def VGG_init(cfgs: dict, model_type: str = 'vgg16') -> VGG:
    if model_type == 'vgg11':
        model = VGG(cfgs['A'], batch_norm=False)
    elif model_type == 'vgg13':
        model = VGG(cfgs['B'], batch_norm=False)
    elif model_type == 'vgg16':
        model = VGG(cfgs['D'], batch_norm=False)
    elif model_type == 'vgg19':
        model = VGG(cfgs['E'], batch_norm=False)

    elif model_type == 'vgg11_bn':
        model = VGG(cfgs['A'], batch_norm=True)
    elif model_type == 'vgg13_bn':
        model = VGG(cfgs['B'], batch_norm=True)
    elif model_type == 'vgg16_bn':
        model = VGG(cfgs['D'], batch_norm=True)
    elif model_type == 'vgg19_bn':
        model = VGG(cfgs['E'], batch_norm=True)

    else:
        print("input error! no such vgg models!\n this program will return vgg16 by default.")
        model = VGG(cfgs['D'], batch_norm=False)
    return model

