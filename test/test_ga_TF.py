import random
import unittest

import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers
import numpy as np

import aikit.ga.TF


class Optimizer(unittest.TestCase):
    def _run(self, params, gradients, optimizer):
        params = [K.variable(param) for param in params]
        placeholders = [K.placeholder(shape=params[i].shape)
                        for i in range(len(params))]

        with aikit.ga.keras.hooks.get_gradients(optimizer, placeholders):
            # 'loss' argument is unnecessary because of our hook
            updates = optimizer.get_updates(None, params)

        for step in range(len(gradients)):
            K.get_session().run(
                updates,
                feed_dict={placeholders[i]: gradients[step][i]
                           for i in range(len(params))}
            )

        return K.get_session().run(params)

    def _test(self, name):
        count = random.randint(2, 10)
        steps = random.randint(2, 10)
        shape = [random.randint(2, 5) for _ in range(2, 5)]

        # numpy generates float64 values by default, whereas TF uses float32 by default
        # we explictly generate float32 values because passing 64-bit floats to TF causes floating-point issues
        params = [np.random.random(shape).astype(np.float32)
                  for _ in range(count)]
        gradients = [[np.random.random(shape).astype(
            np.float32) for _ in range(count)] for _ in range(steps)]

        them = self._run(
            params,
            # running a single step with the reduced (sum) gradients
            [[np.array([gradients[step][i] for step in range(steps)]).sum(
                axis=0) / steps for i in range(count)]],
            getattr(optimizers, name)()
        )

        wrapped = aikit.ga.keras.optimizers.Optimizer(
            getattr(optimizers, name)(), steps=steps)
        self.assertEqual(wrapped.__class__.__name__, 'Optimizer')

        specific = getattr(aikit.ga.keras.optimizers, name)(steps=steps)
        self.assertEqual(specific.__class__.__name__, name)

        for optimizer in [wrapped, specific]:
            # optimizer.optimizer is the Keras optimizer
            self.assertEqual(optimizer.optimizer.__class__.__name__, name)

            us = self._run(
                params,
                gradients,
                optimizer
            )

            self.assertTrue(np.allclose(us, them))

    def testSGD(self):
        self._test('SGD')

    def testRMSprop(self):
        self._test('RMSprop')

    def testAdagrad(self):
        self._test('Adagrad')

    def testAdadelta(self):
        self._test('Adadelta')

    def testAdam(self):
        self._test('Adam')

    def testAdamax(self):
        self._test('Adamax')

    def testNadam(self):
        self._test('Nadam')


if __name__ == '__main__':
    unittest.main()
