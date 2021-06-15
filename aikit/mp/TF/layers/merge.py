import sys

from .keep import Keep


def _layer(layer):
    setattr(
        sys.modules[__name__],
        layer,
        Keep.create(layer)
    )

[_layer(layer) for layer in [
    'Add',
    'Subtract',
    'Multiply',
    'Average',
    'Maximum',
    'Minimum',
]]


def _method(method):
    setattr(
        sys.modules[__name__],
        method,
        lambda inputs, *args, **kwargs: getattr(sys.modules[__name__], method.capitalize())(*args, **kwargs)(inputs)
    )

[_method(method) for method in [
    'add',
    'subtract',
    'multiply',
    'average',
    'maximum',
    'minimum'
]]