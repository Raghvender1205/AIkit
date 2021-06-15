# AIkit: HPO Support
A Submodule and a helper library for HyperParameter Optimization (HPO) Experiments

## Getting Started
------------
### <b>Prerequisites</b>
AIkit: HPO is dependent on the <code>PyYAML</code> PyPI. So install by using
```python
pip install pyyaml
```

## Usage 
- Import the <code>aikit.hpo</code> package
```python
import aikit.hpo
```
- Init the AIkit: HPO Lib with a path to a directory between all the cluster nodes(NFS Server). Use a unique name for the experiment.

A sub directory will be created on the shared folder using this name
```python
aikit.hpo.init('/path/to/nfs', 'model-hpo')
```
- Decide on the HPO strategy:
    - <b>Random Search</b> - Randomally pick up the hyperparameters values.
    - <b>Grid Search</b> - Pick up the hyperparameter values, iterating through all sets across multiple experi,ements.
```python
strategy = aikit.hpo.Strategy.GridSearch
```
- Call the AIkit: HPO Library to specify a set of hyperparameters and pick a specific configuration for the experiment
```python
config = aikit.hpo.pick(
    grid=dict(
        batch_size=[32, 64, 128],
        lr = [1, 0.1, 0.01, 0.001]),
    strategy=strategy)
```
- Use the returned configuration in your code:
```python
optimizer = tf.keras.optimizers.SGD(lr=config['lr'])
```

- Metrics could be reported and saved in the experiment dir under <code>aikit.yaml</code> using <code>aikit.hpo.export</code>

- Pass the Epoch Number and a directory with metrics to be reported. For Example:
```python
aikit.hpo.export(epoch=5, metrics={'accuracy': 0.87})
```