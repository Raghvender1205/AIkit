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
- 2. <i>Maximum GPU batch size</i> - The maximum batch size that can be run in a single GPU in terms of memory. This will be used by the AIkit <code>elastic</code> module for determining whether to use GA or not, and for how many steps. 

Call the <code>init()</code> method 
```python 
aikit.elastic.Torch.init(256, 64)
```
This indicates that the desired global batch size is 256, but a maximum of batch size 64 can be fit into a single GPU.
AIkit will determine according to the available things on how to acheive desired <code>global_batch_size</code>. 
- It would use Multi GPU training in case more GPU's are availalable.
- If not then it will Gradient Accumulation <code>aikit.ga</code>.

The following table shows how many GA steps will be used for different scenarios with different number of available GPUs, for the given example:
| GPUs Available | GA Steps |
|----------------|----------|
| 1              | 4        |
| 2              | 2        |
| 4              | N/A      |

### 3. Parameters
After initialization, <code>aikit.elastic</code> has some calculated parameters.
| Name | Value |
|------|-------|
| `steps` | Number of GA steps, or 1 if GA is not needed |
| `batch_size` | The batch size to be used in every step for every GPU |
| `global_batch_size` | The global batch size that was passed to `init()` |
| `gpus` | The number of GPUs that are used |
| `master` | Mainly relevant for Horovod. `True` if this is the master process (rank 0). |

When passing the <code>batch_size</code> to any Framework use <code>aikit.elastic.batch_size</code>.


#### <b>TensorFlow</b>
For TensorFlow, wrap the <code>Model</code> object with <code>aikit.elastic.TF.model.Model</code>. For example:
```python
model = aikit.elastic.models.Model(model)
``` 
#### <b>PyTorch</b>
For PyTorch, wrap the Model <code>Optimizer</code> with GA. Use <code>aikit.elastic.steps</code> for the number of steps to be accumulated - the value of <code>steps</code> argument of AIkit GA Optimizer. For example:
```python
optimizer = aikit.ga.Torch.optim.Optimizer(optimizer, aikit.elastic.steps)
```

Also, do a Data Parallelise for your model. Use the built-in method of PyTorch.
```python
model = torch.nn.DataParallel(model)
```