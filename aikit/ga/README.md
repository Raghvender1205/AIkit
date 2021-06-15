# AIkit: Gradient Accumulation
A generic Gradient Accumulation wrapper for TF and PyTorch optimizers

## References
* [The problem of batch sizing and limited GPU memory](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce?source=friends_link&sk=74a7a2793da909c1194c0add818c7fd3)
* [What is Gradient Accumulation and how does it help?](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa?source=friends_link&sk=28226e1d0ffa7e450d7dffa8d5b9cff6)
* [How-to guide to using the gradient accumulation mechanism and how we implemented it](https://towardsdatascience.com/how-to-easily-use-gradient-accumulation-in-keras-models-fa02c0342b60?source=friends_link&sk=ff9137c1c7fa5bbfc4c4e09bacc0273b)

## Usage
- First, import GA submodule according to the Framework you are using

1. If you are using TensorFlow, then
```python
import aikit.ga.TF
```
2. If you are using PyTorch, then
```python
import aikit.ga.Torch
```

- Now <b>Wrap an existing optimizer with Gradient Accumulation</b>

In case of having an instance of a TF or PyTorch optimizer or a custom made optimizer, then wrap the <code>optimizer</code> with GA.

TensorFlow
```python
# For TensorFlow
optimizer = aikit.ga.TF.optimizers.Optimizer(optimizer, steps=STEPS)
```

PyTorch
```python
# For PyTorch
optimizer = aikit.ga.Torch.optim.Optimizer(optimizer, steps=STEPS)
```

- <b>Create a gradient-accumulated common optimizer</b>

Create an instance of any common optimizer, already wrapped with Gradient Accumulation.
Just Create an instance of the selected optimizer from AIkit.

Import the optimizer either from <code>aikit.ga.TF.optimizers</code> or <code>aikit.ga.torch.optim</code> For Example:

To Create <code>Adam</code> TF optimizer, wrapped with GA use
```python
optimizer = aikit.ga.TF.optimizers.Adam(steps=STEPS)
```
Both ways require an argument <code>steps</code> to indicate the number of steps to accumulate gradients over.