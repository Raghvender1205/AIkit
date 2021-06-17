import numpy as np
import os
import tempfile
import argparse
import time

import aikit
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

parser = argparse.ArgumentParser(description='Run model parallelism on MNIST.')
parser.add_argument('--splits', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--img_size', type=int, default=400)

args = parser.parse_args()
print(args)

if args.splits > 1:
    aikit.mp.init(splits=args.splits, method=aikit.mp.Method.Cout)

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the TensorFlow backend,'
                       ' because it requires the Dataset API, which is not'
                       ' supported on other platforms.')

class StepTimeReporter1(tf.keras.callbacks.Callback):
    def __init__(self):
        self.start = 0
    
    def on_batch_begin(self, batch, logs={}):
        self.start = time.time()
    
    def on_batch_end(self, batch, logs={}):
        print('>> Step % d took % g sec' % (batch, time.time() - self.start))

class StepTimeReporter(tf.keras.callbacks.Callback):
    def __init__(self):
        self.fname='mnist.txt'
        if (os.path.isfile(self.fname)):
            os.remove(self.fname)
        self.start = 0
        self.saved=False

    def on_batch_begin(self, batch, logs={}):
        #if(os.path.isfile(self.fname))
        if batch == 20 and not self.saved :
            with open(self.fname, 'w') as f:
                f.write(str(tf.get_default_graph().as_graph_def()))
                self.saved=True
        self.start = time.time()

    def on_batch_end(self, batch, logs={}):
        print(' >> Step %d took %g sec' % (batch, time.time() - self.start))


def cnn_layers(inputs):
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    predictions = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return predictions


batch_size = args.batch_size
buffer_size = 1000
steps_per_epoch = int(np.ceil(60000 / float(batch_size)))  # = 469
epochs = 5
num_classes = 10
lr = 2e-3 * batch_size / 128.
img_size = args.img_size

def modify(x, y):
    return tf.squeeze(tf.image.resize_billinear(tf.expand_dins(x, 0), [img_size, img_size]), 0), y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255
x_train = np.expand_dims(x_train, -1)
y_train = tf.one_hot(y_train, num_classes)

# Create the dataset and its associated one-shot iterator.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat()

dataset = dataset.map(modify)


dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()

# Model creation using tensors from the get_next() graph node.
inputs, targets = iterator.get_next()
model_input = layers.Input(tensor=inputs)
model_output = cnn_layers(model_input)
train_model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

train_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[targets])
train_model.summary()

time_reporter = StepTimeReporter()
callbacks = [time_reporter]
train_model.fit(epochs=epochs,
                steps_per_epoch=steps_per_epoch, callbacks=callbacks)

# Save the model weights.
weight_path = os.path.join(tempfile.gettempdir(), 'saved_wt.h5')
train_model.save_weights(weight_path)

# Clean up the TF session.
K.clear_session()

# Second session to test loading trained model without tensors.
x_test = x_test.astype(np.float32)
x_test = np.expand_dims(x_test, -1)

x_test_inp = layers.Input(shape=x_test.shape[1:])
test_out = cnn_layers(x_test_inp)
test_model = tf.keras.models.Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights(weight_path)


test_model.compile(optimizer='rmsprop',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(x_test, y_test, num_classes)
print('\nTest accuracy: {0}'.format(acc))
