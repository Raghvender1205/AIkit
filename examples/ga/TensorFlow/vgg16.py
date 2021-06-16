import tensorflow.keras as tf

import aikit.ga.TF

(x_train, y_train), (x_test, y_test) = tf.datasets.cifar10.load_data()

x_train = tf.applications.vgg16.preprocess_input(x_train)
x_test = tf.applications.vgg16.preprocess_input(x_test)

y_train = tf.utils.to_categorical(y_train)
y_test = tf.utils.to_categorical(y_test)

optimizer = tf.optimizers.SGD()

# Wrap the tf.Optimizer with GA of 2 steps
optimizer = aikit.ga.TF.optimizers.Optimizer(optimizer, steps=2)

model = tf.applications.vgg16.VGG16(
    include_top=True, input_shape=(32, 32, 32),
    weights=None, classes=10)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(x=x_train,
    y=y_train,
    batch_size=64,
    epochs=1,
    verbose=1,
    validation_data=(x_test, y_test),)