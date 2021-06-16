import os
import unittest

import tensorflow
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import aikit.utils
import aikit.reporter
import aikit.reporter.TF

NUM_CLASSES = 10
BATCH_SIZE = 16
STEPS_PER_EPOCH = 1


class MockReporter(aikit.utils.Hook):
    def __init__(self, methodName):
        super(MockReporter, self).__init__(aikit.reporter, methodName)
        self.reported = []

    def __hook__(self, *args, **kwargs):
        reportedInput = args[0]
        wasCurrentInputAlreadyReported = reportedInput not in self.reported
        if wasCurrentInputAlreadyReported:
            self.reported.append(reportedInput)


class TFStopModelCallback(tensorflow.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        self.model.stop_training = True


class TFAutologTest(unittest.TestCase):
    def _run_test(self, run_fit=True, expected_metrics=[], expected_parameters=[]):
        self._mock_env_variables()

        with MockReporter('reportMetric') as reportMetricMock, MockReporter('reportParameter') as reportParameterMock:
            if run_fit:
                self._run_model_with_fit()
            else:
                self._run_model_with_fit_generator()

            self.assertEqual(reportMetricMock.reported,
                             expected_metrics, 'Reported Metrics unmatched')
            self.assertEqual(reportParameterMock.reported,
                             expected_parameters, 'Reported Paramters unmatched')

    def _mock_env_variables(self):
        os.environ["podUUID"] = "podUUId"
        os.environ["reporterGatewayURL"] = "reporterGatewayURL"

    def _run_model_with_fit(self):
        x_train, y_train = self._get_x_train_y_train()
        model = self._create_model_and_compile()

        model.fit(x_train, y_train, batch_size=BATCH_SIZE, callbacks=[
                  TFStopModelCallback()])

    def _run_model_with_fit_generator(self):
        x_train, y_train = self._get_x_train_y_train()
        model = self._create_model_and_compile()
        datagen = ImageDataGenerator()
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                            steps_per_epoch=STEPS_PER_EPOCH)

    def _create_model_and_compile(self):
        model = Sequential()
        model.add(tensorflow.keras.layers.Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        optimizer = tensorflow.keras.optimizers.Adam()

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def _get_x_train_y_train(self):
        (x_train, y_train), (_x_test, _y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = to_categorical(y_train, NUM_CLASSES)
        return x_train, y_train

    def testFitWithoutAutoLog(self):
        self._run_test()

    def testFitWithAutoLog(self):
        aikit.reporter.keras.autolog()
        expected_metrics = ['overall_epochs', 'batch_size',
                            'number_of_layers', 'epoch', 'step', 'accuracy', 'loss']
        expected_parameters = ['optimizer_name', 'learning_rate']
        self._run_test(expected_metrics=expected_metrics,
                       expected_parameters=expected_parameters)
        aikit.reporter.keras.disableAutoLog()

    def testFitAllMetrics(self):
        aikit.reporter.keras.autolog(loss_method=True, epsilon=True)
        expected_metrics = ['overall_epochs', 'batch_size',
                            'number_of_layers', 'epoch', 'step', 'accuracy', 'loss']
        expected_parameters = ['loss_method',
                               'optimizer_name', 'learning_rate', 'epsilon']
        self._run_test(expected_metrics=expected_metrics,
                       expected_parameters=expected_parameters)
        aikit.reporter.keras.disableAutoLog()

    def testFitGeneratorWithoutAutoLog(self):
        self._run_test(run_fit=False)

    def testFitGeneratorWithAutoLog(self):
        aikit.reporter.keras.autolog()
        expected_metrics = [
            'overall_epochs', 'number_of_layers', 'epoch', 'step', 'accuracy', 'loss']
        expected_parameters = ['optimizer_name', 'learning_rate']
        self._run_test(run_fit=False, expected_metrics=expected_metrics,
                       expected_parameters=expected_parameters)
        aikit.reporter.keras.disableAutoLog()

    def testFitGeneratorAllMetrics(self):
        aikit.reporter.keras.autolog(loss_method=True, epsilon=True)
        expected_metrics = [
            'overall_epochs', 'number_of_layers', 'epoch', 'step', 'accuracy', 'loss']
        expected_parameters = ['loss_method',
                               'optimizer_name', 'learning_rate', 'epsilon']
        self._run_test(run_fit=False, expected_metrics=expected_metrics,
                       expected_parameters=expected_parameters)
        aikit.reporter.keras.disableAutoLog()

    #TODO: Add tests that will add new metrics to the compile method and verify they were added.


def pid_exists(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


class TFReporterTest(TFAutologTest):
    def testCreationScope(self):
        with aikit.reporter.keras.Reporter() as reporter:
            self.assertTrue(pid_exists(reporter.pid))

        self.assertFalse(pid_exists(reporter.pid))

    def testAutologCreation(self):
        for autolog in [True, False]:
            with aikit.reporter.keras.Reporter(autolog=autolog) as reporter:
                self.assertEquals(aikit.reporter.keras.autologged(), autolog)

            self.assertFalse(aikit.reporter.keras.autologged())

    def testAutologManual(self):
        with aikit.reporter.keras.Reporter() as reporter:
            self.assertFalse(aikit.reporter.keras.autologged())
            reporter.autolog()
            self.assertTrue(aikit.reporter.keras.autologged())

        self.assertFalse(aikit.reporter.keras.autologged())

    def testAutologSanity(self):
        class Reporter(aikit.reporter.TF.Reporter):
            def __init__(self, *args, **kwargs):
                super(Reporter, self).__init__(*args, **kwargs)
                self.metrics = set()
                self.parameters = set()

            def reportMetric(self, name, value):
                self.metrics.add(name)

            def reportParameter(self, name, value):
                self.parameters.add(name)

        with Reporter() as reporter:
            reporter.autolog()

            self._run_model_with_fit()

            self.assertEqual(reporter.metrics, {'overall_epochs', 'batch_size', 'number_of_layers', 'epoch', 'step', 'accuracy', 'loss'})
            self.assertEqual(reporter.parameters, {'optimizer_name', 'learning_rate'})


if __name__ == '__main__':
    unittest.main()
