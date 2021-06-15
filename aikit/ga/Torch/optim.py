import sys
import torch
import torch.optim as optim
import aikit.utils

class State:
    def __init__(self, steps):
        self.steps = steps
        self.step = 0

# This is an inspired design from Horovod's `DistributedOptimizer`
def Optimizer(optimizer, steps):
    '''
    Wraps any valid PyTorch optimizer with Gradient Accumulation
    '''
    class _GradientAccumulationOptimizer(optim.Optimizer):
        def __init__(self, steps, params):
            self.__class__.__optimizer__.__init__(self, params) # TODO: Defaults -> ? (Add Defaults...!)
            self.state['aikit'] = State(steps)

            aikit.utils.log.debug('Wrapping \'%s\' PyTorch Optimizer with Gradient Accumulation of %d steps', optimizer.__class__.__name__, steps)
        
        def step(self, closure=None):
            loss = None

            if self.aikit.step % self.aikit.steps == self.aikit.steps - 1:
                loss = self.__class__.__optimizer__.step(self, closure)
            else:
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure() # TODO: Make it better ....!

            self.aikit.step += 1
            return loss
        
        def zero_grad(self):
            if self.aikit.step % self.aikit.steps == 0:
                self.__class__.__optimizer__.zero_grad(self)

        @property
        def aikit(self):
            return self.state['aikit']
    
    # the main idea is to dynamically create a class that has all the functionality of the passed optimizer
    # (this is done by inheriting it) while overriding `step()` and `zero_grad()` to accumulate the gradients
    # and actually assign and zero them once in a few steps
    d = dict(_GradientAccumulationOptimizer.__dict__)
    d['__optimizer__'] = optimizer.__class__

    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        d
    )
    
    return cls(steps, optimizer.param_groups)

# declare a GA version of builtin optimizers
def _optimizer(optimizer):
    setattr(
        sys.modules[__name__],
        optimizer,
        lambda steps, *args, **kwargs: Optimizer(getattr(torch.optim, optimizer)(*args, **kwargs), steps)
    )

[_optimizer(optimizer) for optimizer in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'SGD', 'Rprop', 'RMSprop']]