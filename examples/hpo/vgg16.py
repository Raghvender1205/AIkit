import sys
import tensorflow.keras
import tensorflow as tf


# AIKit HPO 
import aikit.hpo

def usage():
    print('usage: python %s <HPO Dir> [strategy: random | grid; default: grid]' % sys.argv[0])
    exit(1)

if len(sys.argv) != 2 and len(sys.argv) != 3:
    usage()

hpo_dir = sys.argv[1]
print('Using HPO dir %s' % hpo_dir)


if len(sys.argv) == 3:
    if sys.argv[2] == 'grid':
        strategy = aikit.hpo.Strategy.GridSearch
    elif sys.argv[2] == 'random':
        strategy = aikit.hpo.Strategy.RandomSearch
    else:
        usage()
else:
    strategy = aikit.hpo.Strategy.GridSearch

# Init the AIkit HPO 
aikit.hpo.init(hpo_dir)

# pick a configuration for this HPO experiment
# we pass the options of all hyperparameters we want to test
# `config` will hold a single value for each parameter
config = aikit.hpo.pick(
    grid=dict(
        batch_size=[32, 64, 128],
        lr = [1, 0.1, 0.01, 0.001]),
    strategy=strategy)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = tf.keras.applications.vgg16.preprocess_input(x_train)
x_test = tf.keras.application.vgg16.preprocess_input(x_test)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

optimizer = tf.keras.optimizers.SGD(lr=config['lr'])

model = tf.keras.application.vgg16.VGG16(
    include_top=True,
    weights=None,
    input_shape=(32,32,3),
    classes=10,
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

class ReportCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        aikit.hpo.report(epoch, dict(
            loss=float(logs['loss']),
            acc=float(logs['acc']),
            val_loss=float(logs['val_loss']),
            val_acc=float(logs['val_acc']),
        ))

model.fit(
    x=x_train,
    y=y_train,
    batch_size=config['batch_size'],
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[ReportCallback()],
)