# horovodrun -np `nvidia-smi --list-gpus | wc -l` -H localhost:`nvidia-smi --list-gpus | wc -l` python examples/elastic/keras/mnist.py

from __future__ import print_function

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import aikit.elastic.TF

aikit.elastic.keras.init(global_batch_size=128, max_gpu_batch_size=16)

NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# wrap 'model' with Run:AI elasticity
model = aikit.elastic.keras.models.Model(model)

model.compile(
    loss='categorical_crossentropy',
    optimizer=tensorflow.keras.optimizers.Adadelta(
        lr=1.0),  # pass any valid Keras optimizer
    metrics=['accuracy']
)

model.fit(x_train, y_train,
          # use the calculated configuration (batch size in this case)
          batch_size=aikit.elastic.batch_size,
          epochs=1,
          verbose=aikit.elastic.master,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=aikit.elastic.master)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
