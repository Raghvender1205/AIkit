# AIkit Python library

Public functional modules for TensorFlow and PyTorch
### Modules

This library consists of a few pretty much independent submodules:

| Module | Name | Info |
|--------|------|------|
| Elastic | `elastic` | [Make Keras and PyTorch models elastic](aikit/elastic/README.md) |
| Gradient Accumulation | `ga` | [Gradient accumulation for Keras and PyTorch optimizers](aikit/ga/README.md) |
| HPO | `hpo` | [Hyperparameter optimization assistance](aikit/hpo/README.md) |
| Model Parallelism | `mp` | Model-parallelism support for Keras builtin layers |
| Auto Profiler | `profiler` | Export timeline of TF/Keras models easily |
| Reporter | `reporter` | [An interface to send metrics and parameters to Promethues Push Gateway](aikit/reporter/README.md) |

### Running The Tests

All tests (unit tests) can be run using the following command:

```
python -m unittest discover -s tests -v
```

## IMPORTANT:
This package is mainly a practice of how to create your own wrapper

### Many things are learned from this

- Multi GPU Training
-  Logging
- About Horovod (https://github.com/horovod/horovod)
- Gradient Accumulation
- How to make a wrapper for different ML Frameworks.


### References
A very good and and a veru helpful repository from which i inspired this repo

<b>Run:AI Python</b>

https://github.com/run-ai/runai