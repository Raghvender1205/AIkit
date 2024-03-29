import tensorflow.keras
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import random

import tensorflow.keras.utils
import tensorflow.keras.backend as K
import tensorflow as tf

import time
import horovod.tensorflow as hvd
import aikit.ga.TF.optimizers

# Horovod: initialize Horovod
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

class StepTimeReporter(tf.keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs={}):
        self.start = time.time()

    def on_batch_end(self, batch, logs={}):
        print(' >> Step %d took %g sec ' % (batch, time.time() - self.start))
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs):
        print(' >> Epoch %d took %g sec time %g sec' % (epoch, time.time() - self.epoch_start, time.time()))

def run(epochs, batch_size, lr, gpus, data_augmentation, cpu_relocation, cpu_merge):
    time_reporter = StepTimeReporter()

    # img_arr is of shape (n, h, w, c)
    def resize_image_arr(img_arr):
        x_resized_list = []
        for i in range(img_arr.shape[0]):
            img = img_arr[i]
            resized_img = resize(img_arr, (224, 224))
            x_resized_list.append(resized_img)
        return np.stack(x_resized_list)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train[:2048]
    y_train = y_train[:2048]
    x_test = x_test[:2048]
    y_test = y_test[:2048]

    print("Resizing images")

    # Resize image arrays
    x_train = resize_image_arr(x_train)
    x_test = resize_image_arr(x_test)

    NUM_CLASSES = 10

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    # Normalize the data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model = ResNet50(include_top=True, weights=None, classes=10)

    STEPS = 2

    if random.choice([True, False]):
        opt = tf.keras.optimizers.SGD(lr=lr)
        opt = aikit.ga.TF.optimizers.Optimizer(opt, STEPS)
    else:
        opt = aikit.ga.TF.optimizers.SGD(STEPS, lr=lr)
    
    opt = hvd.DistributedOptimizer(opt)
    # Not needed to change the device scope for model definition:

    model.compile(loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    if not data_augmentation:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False)
    else:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                                         steps_per_epoch=int(10 * x_train.shape[0] / batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=1,
                            callbacks=[time_reporter,hvd.callbacks.BroadcastGlobalVariablesCallback(0)])
    
    print("BYE BYE")

if __name__ == "__main__":
    run(epochs=1, batch_size=64, lr=1e-3 * hvd.size(), data_augmentation=True, cpu_relocation=False, cpu_merge=True)