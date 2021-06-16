import enum
from . import TF
import aikit.utils
import sys
import tensorflow.keras.layers
from .TF import layers

class Method(enum.Enum):
    Cin = 0
    Cout = 1

def init(splits, method):
    if method != Method.Cin and method != Method.Cout:
        raise ValueError('Unrecognized method %s' % method)
    
    aikit.utils.log.info('Initializing MP (%s) with %s splits', method.name, splits)

    setattr(sys.modules[__name__], 'splits', splits)
    setattr(sys.modules[__name__], 'method', method)

    [setattr(tensorflow.keras.layers, attribute, getattr(layers, attribute)) for attribute in [
        'Activation',
        'add',
        'Add',
        'average',
        'Average',
        'AveragePooling2D',
        'BatchNormalization',
        'Conv2D',
        'Dense',
        'Dropout',
        'Flatten',
        'GlobalAveragePooling2D',
        'GlobalMaxPooling2D',
        'maximum',
        'Maximum',
        'MaxPooling2D',
        'minimum',
        'Minimum',
        'multiply',
        'Multiply',
        'subtract',
        'Subtract',
        'ZeroPadding2D',
    ]]
