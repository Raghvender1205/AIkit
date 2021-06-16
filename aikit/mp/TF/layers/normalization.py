import tensorflow as tf
import tensorflow.keras.layers as layers

import aikit.mp
import aikit.utils

from . import coordinator
from .parallelized import Parallelized

ATTRIBUTES = ['gamma', 'beta', 'moving_mean', 'moving_variance']

class BatchNormalization(Parallelized, layers.BatchNormalization):
    '''
    A MP-supported implementation for tf.keras.layers.BatchNormalization

    The implementation is highly coupled with tf.keras version, and was 
    implemented in regards of TF == 2.4.1

    We rely on the fact that the specified attributes are declared in build(), 
    and are used only by call().
    Instead of implementing the layer ourselves,
    we use the original implementation and cause it for the parallelisation.
    '''
    def build(self, input_shape):
        def add_weight(name, shape, *args, **kwargs):
            '''
            This implementation will be called every time the original build()
            method will create a weight variable. Instead, we make it create
            multiple variables with smaller channel size and placed on different GPUs. 
            '''
            dim = shape[0] // aikit.mp.splits # all weights are of shape (c,)
            return self.add_weight(name, (dim,), *args, *kwargs)

        with aikit.utils.Hook(self, 'add_weight', add_weight, recursion=False): # must not allow recursion as we call add_weight() ourselves
            super(BatchNormalization, self).build(input_shape)
        
        # Every attribute declared by build() is actually a list of Tensors,
        # and not a single one. Make sure they are not accessed by anyone by renaming them
        for attribute in ATTRIBUTES:
            aikit.utils.attribute.rename(self, attribute, '_' + attribute)
    
    def call(self, inputs, *args, **kwargs):
        assert not isinstance(inputs, (tuple, list))

        # We expect all inputs to have underlying parallelized tensors
        # we do not support the other cases as we have already built the 
        # weights for the parallelized version
        assert coordinator.registered(inputs)

        aikit.utils.log.info('Using parallelized input for \'%s\ layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))
        def output(gpu, input):
            values = [getattr(self, '_' + attribute) for attribute in ATTRIBUTES]
            values = [None if value is None else value[gpu] for value in values]

            with aikit.utils.Attribute(self, ATTRIBUTES, values):
                with tf.device('/device:GPU:%d' % gpu):
                    return super(BatchNormalization, self).call(input, *args, **kwargs)
        
        outputs = [output(gpu, input) for gpu, input in enumerate(coordinator.resolve(inputs))]
        return self.merge(outputs, channel_axis=self.axis)