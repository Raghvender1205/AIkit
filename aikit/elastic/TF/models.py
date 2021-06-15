import tensorflow.keras as keras

import aikit.elastic
import aikit.ga.TF
import aikit.utils

def compile(self, *args, **kwargs):
    if 'optimizer' in kwargs:
        assert len(args) == 0
        optimizer = kwargs['optimizer']
    else:
        assert len(args) == 1
        optimizer = args[0]
    
    # TODO: Support cases when 'optimizer' is not a class instance but is either a str or dict.
    if not isinstance(optimizer, keras.optimizers.Optimizer):
        raise ValueError("'optimizer' must be a valid tf.keras.optimer.Optimizer. ")
    
    aikit.utils.log.debug(' compile() called with optimzer %s', optimizer)

    if aikit.elastic.gpus > 1:
        aikit.utils.log.debug('Wrapping optimizer with Horovod')
        import horovod as hvd
        optimizer = hvd.DistributedOptimizer(optimizer)
    
    if aikit.elastic.steps > 1:
        optimizer = aikit.ga.tf.optimizers.Optimizer(optimizer, aikit.elastic.steps)
    kwargs['optimizer'] = optimizer
    
    return self.__aikit__['compile'](**kwargs) # ignore 'args' as 'optimizer' is the only possible args and it is in 'kwargs'

def fit(self, *args, **kwargs):
    aikit.utils.log.debug('fit() called.')

    if aikit.elastic.gpus > 1:
        import horovod.keras as hvd
        callbacks = [hvd.callbacks.BrodcastGlobalVariablesCallback(0)]

        if 'callbacks' in kwargs:
            callbacks.extend(kwargs['callbacks'])

        kwargs['callbacks'] = callbacks

    return self.__aikit__['fit'](*args, **kwargs)


def fit_generator(self, *args, **kwargs):
    aikit.utils.log.debug('fit_generator() called')

    if aikit.elastic.gpus > 1:
        import horovod.keras as hvd
        callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

        if 'callbacks' in kwargs:
            callbacks.extend(kwargs['callbacks'])

        kwargs['callbacks'] = callbacks

    return self.__aikit__['fit_generator'](*args, **kwargs)


def Model(model):
    __aikit__ = {}

    # TODO(levosos): what about evaluate() and predict()?
    for method in [compile, fit, fit_generator]:
        __aikit__[method.__name__] = getattr(model, method.__name__)
        setattr(model, method.__name__, method.__get__(model))

    setattr(model, '__aikit__', __aikit__)

    return model
