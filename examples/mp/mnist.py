import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import aikit.mp # Model Parallelism

aikit.mp.init(splits=2, method=aikit.mp.Method.Cin)

NUM_CLASSES = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential([
    Dense(512, activation='relu', input_shape=(784, )),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax'),
])

optimizer = Adam()
model.compile(loss=tf.keras.losses.categorical_crossentropy(), optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=10,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])
