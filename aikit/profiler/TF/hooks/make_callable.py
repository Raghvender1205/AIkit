import tensorflow as tf

from aikit.utils import Hook

class MakeCallable(Hook):
    def __init__(self):
        super(MakeCallable, self).__init__(tf.Session, '_make_callable_from_options')

    def __hook__(self, _self, callable_options):
        assert not callable_options.HasField('run_session') #TODO: support this

        #TODO: Create a copy of 'callable_options'
        callable_options.run_options.trace_level = tf.RunOptions.FULL_TRACE

        return self.__original__(_self, callable_options)