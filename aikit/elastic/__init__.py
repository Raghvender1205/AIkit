import sys
import aikit.utils

# users should not call this method directly, but call the `init()` method of
# the sub-package of the respecive framework (e.g. `aikit.elastic.keras.init()`)
def __init__(global_batch_size, max_gpu_batch_size, gpus=None):
    if gpus is None:
        gpus = aikit.utils.gpus.count()
    
    if gpus < 1:
        raise ValueError('GPU count (%d) must be atleast 1' % gpus)

    module = sys.modules[__name__]

    setattr(module, 'global_batch_size', global_batch_size)
    setattr(module, 'gpus', gpus)
    setattr(module, 'master', True)

    # TODO(levosos): support uneven dividing must be at least 1
    steps = max(1, global_batch_size // (max_gpu_batch_size * gpus))
    batch_size = global_batch_size // (steps * gpus)

    setattr(module, 'steps', steps)
    setattr(module, 'batch_size', batch_size)

    aikit.utils.log.info('Spreading global batch size %d across %d GPU(s) each with %d step(s) of batch size %d', global_batch_size, gpus, steps, batch_size)
    