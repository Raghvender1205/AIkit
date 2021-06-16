import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers

import aikit.mp
from .keep import Keep
from .parallelized import Parallelized

Activation = Keep.create('Activation')

class Dense(Parallelized, layers.Dense):
    def build(self, input_shape):
        assert len(input_shape) == 2 #TODO: support more than 2 dimension

        total_cin = input_shape[-1]
        cin, cout, c = self.calculate_cs(cin=total_cin, cout=self.units)
        
        self.kernels = self.add_weights(
            name='kernel', shape=(cin, cout), 
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint) #TODO: ok.. ?
        
        if self.use_bias:
            self.biases = self.add_weights(
                name='bias', shape=(c,),
                initializer=self.bias_initializer, regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
    
        self.input_spec = layers.InputSpec(ndim=2, axes={-1: total_cin}) #TODO: use 'ndim_min' once supporing more dimensions
    
    def call(self, inputs):
        inputs = self.inputs(inputs, channel_axis=-1)

        outputs = self.parallelize(
            lambda input, kernel: K.dot(input, kernel),
            input, self.kernels)
        
        if aikit.mp.method == aikit.mp.Method.Cin:
            outputs = self.reduce_split(outputs, channel_axis=-1)
        
        if self.use_bias:
            outputs = self.parallelize(
                lambda output, bias: K.bias_add(output, data_format='channels_last'),
                outputs, self.biases)
        
        if self.activation is not None:
            outputs = self.parallelize(
                lambda output: self.activation(output),
                outputs)
        
        return self.merge(outputs, channel_axis=-1)


Dropout = Keep.create('Dropout')

Flatten = Keep.create('Flatten')