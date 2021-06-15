# AIkit Elastic
A Module for transforming static TensorFlow and PyTorch models to be Elastic

## Info 
Elastic jobs automatically set their hyperparameters (e.g. batch size) according to the current available resources. Elastic jobs use multi-GPU training and gradient accumulation (GA) in order to fully utilize the available resources.

## Getting Started
For Elastic jobs and transforming TF and PyTorch Models Multi-GPU and Gradient Accumulation is needed to be used.

### GA
- For the AIkit provides GA-supported optimizers and wrappers for TF and PyTorch
### Multi-GPU
- For Multi-GPU training, <code>Horovod</code> PyPI is required.

## Usage
### 1. Import Package
- If you are using TensorFlow 
```python
import aikit.elastic.TF
``` 
- If you are using PyTorch, then 
```python
import aikit.elastic.Torch
```

### 2. Initialize the Module
After importing, two main arguments are to passed.
- 1. <i>Global batch size</i> - The batch size, using this model might not fit into a Single GPU as in terms of memory, so the <code>aikit.elastic</code> will use GA and Multi-GPU training for it.
- 2. <i>Maximum GPU batch size</i> - The  