import tensorflow as tf
import numpy as np
import datetime
import pickle
import os

from os.path import join as ospj

from swarm import SwarmCallback
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.core import Flatten
# from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler

SLNUM = os.getenv('SLNUM')
SNNUM = os.getenv('SNNUM')
MIN_PEERS = int(os.getenv('MIN_PEERS', '4'))
GROUP_STR = os.getenv('GROUP_STR')
MODEL_NAME = f'EffiNetB2-CIFAR10-SL-{GROUP_STR}'
SWARM_LEARNING = (SLNUM is not None)
# DATA_DIR = os.getenv('DATA_DIR', './')
OUTPUT_DIR = './output'
# DS_PATH = os.getenv('DS_PATH', './cifar10.npz')
DS_PATH = './cifar10.npz'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 5e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    # print('Learning rate: ', lr)
    return lr

def cifar10_load_data(path):
    loaded = np.load(path)
    train_images = loaded['train_images']
    train_labels = loaded['train_labels']
    test_images = loaded['test_images']
    test_labels = loaded['test_labels']
    
    return (train_images, train_labels), (test_images, test_labels)

# CIFAR-10のインポート
(x_train, y_train), (x_test, y_test) = cifar10_load_data(DS_PATH)

# 変数定義
image_size = 32   #画像のサイズ（CIFAR-10は32×32）
input_shape=(image_size,image_size,3)
num_classes = 10 #画像の種類の数（CIFAR-10は10種類）
epochs = 300
batch_size = 32
init_lr = 1e-3
current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
model_name = f'{MODEL_NAME}_{SNNUM}_{SLNUM}'


# 訓練データと評価データの分割
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                    #   test_size=0.2, stratify=y_train, shuffle=True)

y_train = to_categorical(y_train, num_classes)
# y_valid = to_categorical(y_valid, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True,
                            width_shift_range=0.2, zoom_range=0.2, fill_mode='reflect',
                            brightness_range=[0.9, 1.1],
                            featurewise_center=False,  # set input mean to 0 over the dataset
                            samplewise_center=False,  # set each sample mean to 0
                            featurewise_std_normalization=False,  # divide inputs by std of dataset
                            samplewise_std_normalization=False,  # divide each input by its std
                            zca_whitening=False,  # apply ZCA whitening
            )
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    verbose=1,
    min_lr=1e-6,
    min_delta=1e-4,
    mode='min',
    cooldown=3,
)
earlystopping = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,
    patience=40,
    mode='min',
    verbose=1,
    # restore_best_weights=True
)

steps_per_epoch = (x_train.shape[0] + batch_size - 1) // batch_size

real_sync_interval = steps_per_epoch * 1

print("-------------------")
print("SWARM-LERANING!")
print(f"+++ REAL SYNC INTERVAL: {real_sync_interval}")
print(f"+++ MIN PEERS: {MIN_PEERS}")
# print(f"+++ NODE WEIGHTAGE: {NODE_WEIGHTAGE}")
print(f"+++ BATCH SIZE: {batch_size}")
print(f"+++ MODEL NAME: {MODEL_NAME}")
print(f"+++ TF VERSION: {tf.version.VERSION}")
print("-------------------")

swarmCallback = SwarmCallback(
    sync_interval=real_sync_interval,
    min_peers=MIN_PEERS,
    # node_weightage=NODE_WEIGHTAGE,
    val_data=(x_test, y_test),
    val_batch_size=batch_size,
    model_name=MODEL_NAME,
    use_adaptive_sync=False
)

lrschedule = LearningRateScheduler(lr_schedule, verbose=1)
modelcheckpoint = ModelCheckpoint(filepath=ospj(OUTPUT_DIR, f'saved_models/{model_name}/{current_time}/') + '{epoch:02d}.h5',
                                  save_best_only=False, save_weights_only=False)
tensorboard = TensorBoard(log_dir=ospj(OUTPUT_DIR, f'logs/{model_name}/{current_time}/'))

callbacks = [
    # earlystopping,
    # reduce_lr,
    modelcheckpoint,
    tensorboard,
    lrschedule,
    swarmCallback,
]

# lr = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=init_lr,
#     decay_steps=steps_per_epoch,
#     decay_rate=0.97,
# )
lr = init_lr
# optimizer = AdamW(learning_rate=lr, weight_decay=1e-4)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights=None, pooling='avg')

model = tf.keras.Sequential([
    base_model,
    BatchNormalization(),
    Flatten(),
    # Dropout(0.2),
    # Dense(512, activation='relu'),
    # BatchNormalization(),
    # Dropout(0.2),
    # Dense(256, activation='relu'),
    # BatchNormalization(),
    # Dropout(0.2),
    Dense(num_classes, kernel_initializer='he_normal', activation='softmax')
])
# tf.keras.utils.plot_model(model, to_file=f'graphs/{model_name}/{current_time}/model.png', show_shapes=True)

# base_model.trainable = False
# model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(train_generator,
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=pre_epochs,
#                     validation_data=(x_valid, y_valid),
#                     shuffle=True,
#                     # workers=4,
#                     use_multiprocessing=False,
#                     max_queue_size=32,
#                     callbacks=callbacks
#                     )
# model.evaluate(x_test, y_test)


base_model.trainable = True
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks,
                    shuffle=True,
                    # workers=4,
                    use_multiprocessing=False,
                    max_queue_size=32,
                    # initial_epoch=pre_epochs
                    )
# eval_result = model.evaluate(x_test, y_test)
Path(ospj(OUTPUT_DIR, f'results/{model_name}/{current_time}')).mkdir(parents=True, exist_ok=True)
pickle.dump(
    # {
        # 'history':
            history.history,
    #     'evaluation': eval_result,
    # },
    open(ospj(OUTPUT_DIR, f"results/{model_name}/{current_time}/history_with_test_" + current_time + ".pkl"), "wb")
    )
    
# print(eval_result)
print(model_name, current_time)
