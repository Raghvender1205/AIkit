# AIkit: Reporter
An Interface to send metrics and parameters to [Promethues Pushgateway](https://github.com/prometheus/pushgateway)

## Getting Started
----------------
###  Requirements
It is required to have some environment variables:
1. <code>podUUID</code>
2. <code>reporterGatewayURL</code>

These environment variables will be added to each pod when a job was created using the AIkit.

### Reporting Options
1. <i>Parameters</i>: Key-value input parameters of your choice. Key and value are strings.
2. <i>Metrics</i>: Key-value input parameters of your choice. Key is a string. value is numeric.
----------
## Usage
### 1. Import
Import the submodule using the following command:
```python
import aikit.reporter
``` 
If Using TensorFlow, then:
```python
import aikit.reporter.TF
```

### 2. Scoped and Non-Scoped APIs
There are two ways to use AIkit reporting Lib. The first approach is to use a reporter object as a Python context manager (use <code>with</code>) and the second approach is to call non-scoped methods.

The communication with the Pushgateway is done in another process, to reduce the performance impact of this library on the main process, making the reporting itself is done asynchronously. Proper termination is required to ensure everything was successfully reported before the Python process terminates.

Therefore, it is highly recommended to use the <b>scoped API</b> as it shuts down the reporting process properly.

If using the non-scoped API, reports are not guaranteed to be successfully reported without explicit termination (using the <code>finish</code> method) or upon failures.

- <i><b>Scoped API - Recommended</b></i>

First, create a reporter object <code>aikit.reporter.Reporter</code> using a <code>with</code> statement. For Example:
```python
with aikit.reporter.Reporter() as reporter:
    pass
```

Then use <code>reporter</code> to report metrics and parameters using methods <code>reportMetric</code> and <code>reportParameter</code>

When using TF, it is preferable to use <code>aikit.reporter.TF.Reporter</code> instead because TF reporter lets you add auto logging to the <code>TF.keras</code> models.

This can be done by passing <code>autolog=True</code> when creating TF reporter, or by calling its <code>autolog()</code> method. For Example:
```python
with aikit.reporter.TF.Reporter(autolog=True) as reporter:
    pass
```
or
```python
with aikit.reporter.TF.Reporter() as reporter:
    reporter.autolog()
```

- <i><b>Non-Scoped API</b></i>

Use the methods <code>reportMetric</code> and <code>reportParameter</code> from <code>aikit.reporter</code> without creating any objects.

In this, a global <code>Reporter()</code> object is created in background.

Now, for this at the end of the script you need to explicitly call <code>aikit.reporter.finish()</code> .This is to properly terminate the reporter process and ensure everything was successfully reported before the Python script ends.

-----
## API Reference
<code>aikit.reporter.Reporter.reportMetric</code> and <code>aikit.reporter.reportMetric</code>

Sends a metric with a name <i>"reporter_push_gateway_metric_[reporter_metric_name]"</i>

<b>Scoped API</b>
```python
with aikit.reporter.Reporter() as reporter:
    reporter.reportMetric('batch_size', 100)
```

<b>Non-Scoped API</b>
```python
aikit.reporter.reportMetric('batch_size', 100)
```

---

`aikit.reporter.Reporter.reportParameter` and `aikit.reporter.reportParameter`

Sends a parameter with the following name <i>"reporter_push_gateway_metric_[reporter_parameter_name]"</i>.

Scoped API:
```python
with aikit.reporter.Reporter() as reporter:
    reporter.reportParameter('loss_method', 'categorical_crossentropy')
```

Non-scoped API:
```python
aikit.reporter.reportParameter('loss_method', 'categorical_crossentropy')
```

### Scoped API
---

`aikit.reporter.TF.Reporter.autolog`

Enables TF automatic logging. This could be done by passing `autolog=True` when creating the TF reporter, or by calling its `autolog()` method. For example:

```python
with aikit.reporter.TF.Reporter(autolog=True) as reporter:
    pass
```

or

```python
with aikit.reporter.TF.Reporter() as reporter:
    reporter.autolog()
```

### Non-Scoped API

---

`aikit.reporter.finish`

Make sure all reports were sent successfully and gracefully terminate the background process (used for communication with Pushgateway).
Simply add the following line at the very end of your script:

```python
aikit.reporter.finish()
```

---

`aikit.reporter.TF.autolog`

Enables automatic metrics and parameters updates for Keras.
Simply add the following line before running the model:

```python
aikit.reporter.TF.autolog()
```

---

`aikit.reporter.TF.disableAutolog`

Disables automatic metrics and parameters updates from Keras fit, fit_generator methods.
Simply add the following line before running the model

```python
aikit.reporter.TF.disableAutolog()
```