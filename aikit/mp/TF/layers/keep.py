import tensorflow.keras.layers as layers
import tensorflow as tf

import aikit.utils
from . import coordinator

class Keep(layers.Layer):
    '''
    An MP-supported generic TF Keras Layer

    The main goal of this class is to ease the implementation of MP-supported
    TF layers in order to avoid merging an already parallelised layer
    
    In case that the input Tensor is a AIkit parallelised Layer, we duplicate 
    this layer and activate it to every part of the parallelised input layer.
    Otherwise, the Layer is activated on the non-paralledlised input layer,
    just as it was supposed to run originally, and the input tensor is not
    being split. Anyways, the layer actions and algorithm are kept untouched.

    If the input layer is indeed a AIkit parallelised one, the merged tensor
    is the concatenation of the layer's outputs. The channel dimension is taken
    from the 'data_format' attribute, if one exists, otherwise it is assumed to
    be the last dimension.

    The following example shows the essence of the inheritance being done, where:
        1) 'Layer' is the tf.keras.layers.Layer
        2) 'Keras' is the tf.keras implementation of a layer.
        3) 'AIkit' is the very class (aikit.mp.layers.Keep)
        4) 'Final' is the MP-supported tf.keras Layer implementation
    
    class Layer(object):
        def call(self):
            print('Layer.call()')
    
    class Keras(Layer):
        def call(self):
            print('Keras.call()')
            super(Keras, self).call()

    class AIkit(Layer):
        def call(self):
            print('Runai.call()')
            super(Runai, self).call()
    
    class Final(AIkit, Keras): pass
    '''
    def call(self, inputs):
        if isinstance(inputs, (tuple, list)):
            valid = all([coordinator.registered(input) for input in inputs])
        else:
            valid = coordinator.registered(inputs)
            
        if not valid:
            aikit.utils.log.debug('Not Parallelising \'%s\' layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))
            return super(Keep, self).call(inputs)
        
        aikit.utils.log.info('Using Parallelised input for \'%s\ layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))
        channel_axis = 1 if getattr(self, 'data_format', 'channels_last') == 'channels_first' else -1

        def output(gpu, inputs):
            with tf.device('/device:GPU:%d' % gpu):
                return super(Keep, self).call(inputs)
        
        if isinstance(inputs, (tuple, list)):
            inputs = [coordinator.resolve(input) for input in inputs]
            inputs = [[input[i] for input in inputs] for i in range(len(inputs[0]))]
        else:
            inputs = coordinator.resolve(inputs)

        outputs = [output(gpu, input) for gpu in enumerate(inputs)]
        merged = layers.Concatenate(axis=channel_axis)(outputs)
        coordinator.register(merged, outputs)
        return merged
    
    @staticmethod
    def create(name):
        '''
        Create an MP-supported TF Keras layer with 'Keep' logic
        '''