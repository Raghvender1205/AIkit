import aikit.elastic
import aikit.utils

from . import models


def init(global_batch_size, max_gpu_batch_size, gpus=None):
    # first of all calculate the number of GA steps and the batch size
    aikit.elastic._init(global_batch_size, max_gpu_batch_size, gpus)

    # now use Horovod if needed
    if aikit.elastic.gpus > 1:
        aikit.utils.log.debug('Initializing Horovod')
        import horovod.keras as hvd
        hvd.init()

        setattr(aikit.elastic, 'master', hvd.local_rank() == 0)
        # so that anyone will be easily accessible to Horovod
        setattr(aikit.elastic, 'hvd', hvd)

        aikit.utils.log.debug('Attaching Keras session to GPU #%d', hvd.local_rank())
        import tensorflow
        config = tensorflow.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        from tensorflow.keras.backend import set_session 
        #import keras.backend
        # TODO(levosos): support cases where configuration will be set afterwards
        set_session(tensorflow.Session(config=config))
