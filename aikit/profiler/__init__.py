import tensorflow as tf

from .profiler import Profiler
from . import TF

def profile(steps, dst='/tmp/'):
    profile.session_run = TF.hooks.SessionRun(steps, dst)
    profile.session_run.enable()

    if '_Callable' in dir(tf.Session): # TF Keras API
        profile.make_callable = TF.hooks.make_callable()
        profile.run_callable = TF.hooks.run_callable(steps, dst)

        profile.make_callable.enable()
        profile.run_callable.enable()