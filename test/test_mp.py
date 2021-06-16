import random
import unittest

import tensorflow.keras.layers as layers

import aikit.mp
import aikit.utils


class Base(layers.Layer):  # acts as a Keras layer
    def __init__(self):
        self.calls = []

    def call(self, inputs):
        self.calls.append(inputs)


class Layer(aikit.mp.TF.layers.keep.Keep, Base):
    pass


class KeepLayer(unittest.TestCase):
    def testNotRegistered(self):
        # previous layer
        merged = aikit.utils.random.string()

        # tested layer
        layer = Layer()
        layer.call(merged)

        assert layer.calls == [merged]

    def testRegistered(self):
        class init_hook(aikit.utils.Hook):
            def __init__(self):
                super(init_hook, self).__init__(
                    layers.Concatenate, '__init__')

            def __hook__(self, *args, **kwargs):
                if len(args) == 2:
                    self.axis = args[1]
                else:
                    self.axis = kwargs['axis']

                self.__original__(*args, **kwargs)

        class call_hook(aikit.utils.Hook):
            def __init__(self):
                super(call_hook, self).__init__(
                    layers.Concatenate, '__call__')
                self.merge = aikit.utils.random.string()

            def __hook__(self, *args, **kwargs):
                self.inputs = args[1]
                return self.merge

        for data_format, channel_axis in [(None, -1), ('channels_last', -1), ('channels_first', 1)]:
            # previous layer
            merged = aikit.utils.random.string()
            outputs = [aikit.utils.random.string()
                       for _ in range(random.randint(2, 7))]
            aikit.mp.keras.layers.coordinator.register(merged, outputs)

            # tested layer
            with init_hook() as init, call_hook() as call:
                layer = Layer()

                if data_format != None:
                    setattr(layer, 'data_format', data_format)

                merged = layer.call(merged)

                # Keras layer was called with the parallelised input layer
                assert layer.calls == outputs
                assert init.axis == channel_axis
                assert aikit.mp.keras.layers.coordinator.registered(merged)
                assert aikit.mp.keras.layers.coordinator.resolve(
                    merged) == call.inputs


if __name__ == '__main__':
    unittest.main()
