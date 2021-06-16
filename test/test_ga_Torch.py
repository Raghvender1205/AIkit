import unittest
import numpy as np
import pickle
import torch

import aikit.ga.Torch
import aikit.utils.random

class Dry(unittest.TestCase):
    class Optimizer(torch.optim.Optimizer):
        def __init__(self, params=None, defaults=None):
            if params is None:
                params = [torch.empty(aikit.utils.random.shape())]
            
            if defaults is None:
                defaults = dict()

            super(Dry.Optimizer, self).__init__(params=params, defaults=defaults)
            self.log_step = []
            self.log_zero_grad = []

        def step(self, closure=None):
            self.log_step.append(self.aikit.step)

            if closure is not None:
                return closure()

        def zero_grad(self):
            self.log_zero_grad.append(self.aikit.step)

    def testCreation(self):
        optimizer = Dry.Optimizer()
        steps = aikit.utils.random.number()
        wrapped = aikit.ga.torch.optim.Optimizer(optimizer, steps)

        self.assertEqual(wrapped.aikit.steps, steps)
        self.assertEqual(wrapped.aikit.step, 0)

    def testPickle(self):
        optimizer = Dry.Optimizer()
        steps = aikit.utils.random.number()
        wrapped = aikit.ga.torch.optim.Optimizer(optimizer, steps)

        wrapped.aikit.step = aikit.utils.random.number()
        state = wrapped.state_dict()
        self.assertEqual(wrapped.aikit, state['state']['aikit'])
        dumped = pickle.dumps(state)
        loaded = pickle.loads(dumped)
        self.assertEqual(state['state']['aikit'].steps,
                         loaded['state']['aikit'].steps)
        self.assertEqual(state['state']['aikit'].step,
                         loaded['state']['aikit'].step)

    def testStep(self):
        optimizer = Dry.Optimizer()
        steps = aikit.utils.random.number()
        wrapped = aikit.ga.torch.optim.Optimizer(optimizer, steps)

        minimum = steps * 2
        maximum = steps * 100

        iterations = aikit.utils.random.number(minimum, maximum)

        for _ in range(iterations):
            wrapped.step()

        self.assertEqual(wrapped.log_step, list(
            range(steps - 1, iterations, steps)))

    def testStepClosure(self):
        optimizer = Dry.Optimizer()
        steps = aikit.utils.random.number()
        wrapped = aikit.ga.torch.optim.Optimizer(optimizer, steps)

        minimum = steps * 2
        maximum = steps * 100

        iterations = aikit.utils.random.number(minimum, maximum)

        value = aikit.utils.random.number()

        for _ in range(iterations):
            def closure(): return value
            loss = wrapped.step(closure)
            self.assertEqual(loss, value)

    def testZeroGrad(self):
        optimizer = Dry.Optimizer()
        steps = aikit.utils.random.number()
        wrapped = aikit.ga.torch.optim.Optimizer(optimizer, steps)

        minimum = steps * 2
        maximum = steps * 100

        iterations = aikit.utils.random.number(minimum, maximum)

        for _ in range(iterations):
            wrapped.zero_grad()
            wrapped.step()  # must be called to advance `aikit.step`

        self.assertEqual(wrapped.log_zero_grad,
                         list(range(0, iterations, steps)))


class Wet(unittest.TestCase):
    def _run(self, seed, features, xs, ys, optimizer_fn):
        # convert single tensors to tuples
        if not isinstance(xs, tuple):
            xs = (xs,)

        if not isinstance(ys, tuple):
            ys = (ys,)

        # set PyTorch seed to get the same results
        torch.manual_seed(seed)

        # create a model
        model = torch.nn.Linear(in_features=features,
                                out_features=features, bias=False)

        # create an optimizer
        optimizer = optimizer_fn(model.parameters())

        # create a loss function
        # we explicitly use 'sum' reduction and not 'mean' because
        # the number of samples is different when using GA
        loss_fn = torch.nn.MSELoss(reduction='sum')

        for x, y in zip(xs, ys):
            optimizer.zero_grad()
            predictions = model(x)
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()

        return model.weight.detach().np()

    def _run_wo_ga(self, seed, features, name, x, y, **kwargs):
        return self._run(
            seed,
            features,
            x,
            y,
            lambda params: getattr(torch.optim, name)(params=params, **kwargs))

    def _run_ga_wrapped(self, seed, steps, features, name, xs, ys, **kwargs):
        return self._run(
            seed,
            features,
            xs,
            ys,
            lambda params: aikit.ga.torch.optim.Optimizer(getattr(torch.optim, name)(params=params, **kwargs), steps=steps))

    def _run_ga_builtin(self, seed, steps, features, name, xs, ys, **kwargs):
        return self._run(
            seed,
            features,
            xs,
            ys,
            lambda params: getattr(aikit.ga.torch.optim, name)(steps=steps, params=params, **kwargs))

    def _test(self, name, **kwargs):
        seed = aikit.utils.random.number()

        # randomize test parameters
        steps = aikit.utils.random.number(2, 10)
        batch = aikit.utils.random.number(5, 10)
        samples = steps * batch
        features = aikit.utils.random.number(1, 5)
        shape = (samples, features)

        # generate dummy tensors
        x = torch.randn(shape)
        y = torch.randn(shape)

        # split them accordingly
        xs = x.split(batch, dim=0)
        ys = y.split(batch, dim=0)

        assert len(xs) == steps
        assert len(ys) == steps

        wo = self._run_wo_ga(seed, features, name, x, y, **kwargs)
        wrapped = self._run_ga_wrapped(
            seed, steps, features, name, xs, ys, **kwargs)
        builtin = self._run_ga_builtin(
            seed, steps, features, name, xs, ys, **kwargs)

        self.assertTrue(np.allclose(wo, wrapped))
        self.assertTrue(np.allclose(wrapped, builtin))

    def testAdadelta(self):
        self._test('Adadelta')

    def testAdagrad(self):
        self._test('Adagrad')

    def testAdam(self):
        self._test('Adam')

    def testAdamW(self):
        self._test('AdamW')

    def testAdamax(self):
        self._test('Adamax')

    def testASGD(self):
        self._test('ASGD')

    def testSGD(self):
        lr = 0.01 * aikit.utils.random.number(1, 10)
        self._test('SGD', lr=lr)

    def testRprop(self):
        self._test('Rprop')

    def testRMSprop(self):
        self._test('RMSprop')


if __name__ == '__main__':
    unittest.main()
