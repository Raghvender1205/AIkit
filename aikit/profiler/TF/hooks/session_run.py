from aikit.profiler import Profiler
import tensorflow as tf

class SessionRun(Profiler):
    def __init__(self, steps, dst):
        super(SessionRun, self).__init__(tf.Session, 'run', steps, dst)
    
    def __hook__(self, _self, fetches, feed_dict=None, options=None, run_metadata=None):
        if options is None:
            options = tf.RunOptions()
        
        options.trace_level = tf.RunOptions.FULL_TRACE

        if run_metadata is None:
            run_metadata = tf.RunMetadata()

        result = self.__original__(_self, fetches, feed_dict, options, run_metadata)
        self._update(run_metadata)

        return result