import tensorflow.keras.layers as layers
import tensorflow as tf

import aikit.mp
import aikit.utils

from . import coordinator


class Parallelized(layers.Layer):
    '''
    A Helper class for MP-supported layers 
    '''
    def add_weights(self, name, shape, **kwargs):
        '''
        Declare parallelised weights

        Args:
            - name: The base name of the weights
            - shape: The (split) shape of each weight
        
        Returns:
            A list of Tensors with the tensor names of the weights
        '''
        aikit.utils.log.debug('Declaring %d weights (%s) of shape %s for \'%s\' layer "%s"' % (aikit.mp.splits, name, shape, self.__class__.__name__, getattr(self, 'name', 'N/A')))
        
        def add_weight(gpu):
            with tf.device('/device:GPU:%d' % gpu):
                return self.add_weight(
                    name='%s_%d' % (name, gpu),
                    shape = shape, **kwargs)
        
        return [add_weight(gpu) for gpu in range(aikit.mp.splits)]

    def calculate_cs(self, cin, cout):
        '''
        Calculate the 'C' dimensions

        Using the total (original) cin and cout dimensions sizes, calculate the split 
        sizes of those dimensions. In addition, provide an extra dimension size (c) 
        which is the split size of the output dimension

        Args:
            cin: Total cin size
            cout: Total cout size
        Returns:
            A tuple (cin, cout, c) of the per-GPU sizes
        '''
        # TODO: Support uneven Division 
        c = cout // aikit.mp.splits # output dimension is always split 

        if aikit.mp.method == aikit.mp.Method.Cin:
            cin = cin // aikit.mp.splits
        elif aikit.mp.method == aikit.mp.Method.Cout:
            cout = c
        else:
            raise ValueError('Unrecognized MP Method: %s' % aikit.mp.method)

    def inputs(self, input, channel_axis):
        '''
        Get the respective split (per-GPU) inputs for an input tensor

        Returns the split inputs of a merged tensor in the correct way.
        This varies between the MP methods.
        ''' 

        assert isinstance(input, tf.Tensor) # not a list/tuple of Tensors
        
        if aikit.mp.method == aikit.mp.Method.Cin:
            if coordinator.registered(input):
                aikit.utils.log.info('Using Parallelized input for \'%s\' layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))
                return coordinator.resolve(input)
            else:
                aikit.utils.log.warning('Splitting non-parallelized input (%s) for \'%s\' layer "%s"', input.name, self.__class__.__name__, getattr(self, 'name', 'N/A'))
                return tf.split(input, aikit.mp.splits, axis=channel_axis)
        elif aikit.mp.method == aikit.mp.Method.Cout:
            if coordinator.registered(input):
                aikit.utils.log.info('Gathering parallelised input for \'%s\' layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))

                def gather(gpu):
                    with tf.device('/device:GPU:%d' % gpu):
                        return layers.Concatenate(axis=channel_axis)(coordinator.resolve(input))
                    
                return [gather(gpu) for gpu in range(aikit.mp.splits)]
            else:
                return [input] * aikit.mp.splits
        else:
            raise ValueError('Unrecognized MP method: %s' % aikit.mp.method)
    
    def merge(self, outputs, channel_axis):
        '''
        Merge and register the split outputs of a layer

        Merges a parallelized layer's output tensors and register 
        then in coordinator's for later use.
        The merge is done be concatenating the Tensors on the specified channel axis

        Returns:
            A Tensor representing the merged output of the layer
        '''
        merged = layers.Concatenate(axis=channel_axis)
        coordinator.register(merged, outputs)
        return merged
    
    def parallelize(self, l, *iterables):
        '''
        Evaluate a lambda on each GPU with device placement
        '''
        def wrap(gpu, *args):
            with tf.device('/device:GPU:%d' % gpu):
                return l(*args)
        
        return [wrap(gpu, *args) for gpu, args in enumerate(zip(*iterables))]
    
    def reduce_split(self, outputs, channel_axis):
        '''
        Reduce-split some tensors on a specified axis
        '''
        def split(gpu):
            with tf.device('/device:GPU:%d' % gpu):
                return tf.split(outputs[gpu], aikit.mp.splits, axis=channel_axis)
        
        splits = [split(gpu) for gpu in range(aikit.mp.splits)] 

        def output(gpu):
            with tf.device('/device:GPU:%d' % gpu):
                return layers.Add()([split[gpu] for gpu in splits])
        
        return [output(gpu) for gpu in range(aikit.mp.splits)]