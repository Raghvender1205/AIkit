from aikit.profiler import Profiler

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as tf_c_api
from tensorflow.python.util import compat

def RunCallable(Profiler):
    def __init__(self, steps, dst):
        super(RunCallable, self).__init__(tf_c_api, 'TF_SessionRunCallable', steps, dst)
    
    def __hook__(self, session, handle, args, status, run_metadata_ptr):
        assert run_metadata_ptr is None #TODO: handle this

        run_metadata_ptr = tf_c_api.TF_NewBuffer()
        result = self.__original__(session, handle, args, status, run_metadata_ptr)

        proto = tf_c_api.TF_GetBuffer(run_metadata_ptr)
        tf_c_api.TF_DeleteBuffer(run_metadata_ptr)

        run_metadata = tf.RunMetadata()
        run_metadata.ParseFromString(compat.as_bytes(proto))

        self._update(run_metadata)

        return result
