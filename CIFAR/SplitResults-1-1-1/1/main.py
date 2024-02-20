#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.applications.densenet import DenseNet
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Flatten, Dropout
from tensorflow.keras.layers import concatenate, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers
import matplotlib.pyplot as plt
# import configparser
import json
import numpy as np
from tensorflow.keras.utils import to_categorical
import pathlib
import argparse
import sys
import os


# ```bash
# docker container rm sl-$SLNUM ; bash ./swarm-learning/bin/run-sl        \
#     --name=sl-$SLNUM                         \
#     --network sl-net             \
#     --sl-platform=TF                   \
#     --host-ip=sl-$SLNUM                \
#     --sn-ip=sn-$SNNUM                   \
#     --data-dir="/home/yudonghan/storage/NIHCHEST"  \
#     --model-dir="/home/yudonghan/storage/NIHCHEST/SplitResult/$SLNUM"  \
#     --model-program=train.py        \
#     --apls-ip apls                 \
#     --gpu=$GPU                        \
#     -serverAddress spire-server            \
#     -genJoinToken  \
#     -e SLNUM=$SLNUM  \
#     -e SNNUM=$SNNUM  \
#     -e WEIGHTAGE=$WEIGHTAGE
# ```

# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# In[ ]:


@keras_export('keras.applications.densenet.DenseNet49',
              'keras.applications.DenseNet49')
def DenseNet49(include_top=False,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000):
    """Instantiates the Densenet49 architecture."""
    return DenseNet([4, 4, 8, 6], include_top, weights, input_tensor, input_shape, pooling, classes)


# In[ ]:


config = json.load(open('config.json'))
print(config)


# In[ ]:


LR = config['learning_rate']
EPOCHS = config['epochs']
BATCH_SIZE = config['batch_size']
MODEL = config['model']
DROPOUT = config['dropout']
BASE_NAME = config['base_name']
SWARM_LEARNING = config['sl']
MIN_PEERS = config['min_peers']
SYNC_INTERVAL = config['sync_interval']
WEIGHTAGE = os.getenv('WEIGHTAGE', 50)
DS_PATH = 'cifar10.npz'


# In[ ]:


# parser = argparse.ArgumentParser()
# parser.add_argument('-n', '--name', type=str, default=BASE_NAME)
# parser.add_argument('-p', '--path', type=str, default=DS_PATH)
# args = parser.parse_args()
# if args.name:
#     BASE_NAME = args.name
# if args.path:
#     DS_PATH = args.path


# Example Usage:
# 
# ```bash
# python3 main.py -n cifar10-CL-norm -p data/cifar10.npz
# python3 main.py -n cifar10-LL-norm -p SplitResults-1-1-1/1/cifar10.npz
# python3 main.py -n cifar10-LL-norm -p SplitResults-1-1-1/2/cifar10.npz
# python3 main.py -n cifar10-LL-norm -p SplitResults-1-1-1/3/cifar10.npz
# ```

# In[ ]:


loaded = np.load(DS_PATH)
train_images = loaded['train_images']
train_labels = loaded['train_labels']
test_images = loaded['test_images']
test_labels = loaded['test_labels']
train_num = train_images.shape[0]


# In[ ]:


NAME = BASE_NAME + f'_{MODEL}_lr_{LR}_epochs_{EPOCHS}_batch_size_{BATCH_SIZE}_dropout_{DROPOUT}_train_num_{train_num}_W{WEIGHTAGE}'
print(NAME)


# In[ ]:


train_labels = tf.squeeze(tf.one_hot(train_labels, 10))
test_labels = tf.squeeze(tf.one_hot(test_labels, 10))


# In[ ]:


train_images.shape


# In[ ]:


# Normalize pixel values to be between -1 and 1
if config['norm']:
    # mean = config['mean'] if 'mean' in config else train_images.mean()
    # std = config['std'] if 'std' in config else train_images.std()
    # train_images = (train_images - mean) / std
    # test_images = (test_images - mean) / std
    train_images = tf.image.per_image_standardization(train_images)
    test_images = tf.image.per_image_standardization(test_images)


# In[ ]:


print(train_images.shape, train_labels.shape)


# In[ ]:


# print(MODEL)
# if MODEL == 'custom_cnn':
#     model = tf.keras.Sequential()
#     model.add(layers.InputLayer(input_shape=(32, 32, 3)))
#     model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Dropout(DROPOUT))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D((2, 2)))
#     # model.add(layers.Dropout(DROPOUT))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
#     model.add(layers.Dropout(DROPOUT))
#     model.add(layers.Dense(10, activation='softmax'))
#     model.build()
# elif MODEL == 'efficientnetb0':
#     base_model = tf.keras.applications.EfficientNetB0(weights=None, pooling='avg', input_shape=(32, 32, 3))
#     model = tf.keras.Sequential([
#         base_model,
#         layers.Dense(10, activation='softmax')
#     ])
#     model.build(input_shape=base_model.input_shape)
# elif MODEL == 'densenet49':
#     base_model = DenseNet49(weights=None, pooling='avg', input_shape=(32, 32, 3))
#     model = tf.keras.Sequential([
#         base_model,
#         layers.Dense(10, activation='softmax')
#     ])
#     model.build(input_shape=base_model.input_shape)


# In[ ]:


base_model = tf.keras.applications.DenseNet121(weights=None, input_shape=(32, 32, 3), include_top=False, pooling='avg')


# In[ ]:


model = tf.keras.Sequential([
    base_model,
    layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)


# In[ ]:


pathlib.Path('./history').mkdir(parents=True, exist_ok=True)
pathlib.Path('./models').mkdir(parents=True, exist_ok=True)


# In[ ]:


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{NAME}', profile_batch=5),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'./models/{NAME}.h5', save_best_only=False, save_weights_only=False),
    tf.keras.callbacks.LearningRateScheduler(scheduler),
]


# In[ ]:


if SWARM_LEARNING:
    step_per_epoch = (train_num + BATCH_SIZE - 1) // BATCH_SIZE
    from swarm import SwarmCallback
    real_sync_interval = SYNC_INTERVAL  #*step_per_epoch
    NAME += f'_INTERVAL_{real_sync_interval}'

    print("-------------------")
    print("SWARM-LERANING!")
    print(f"+++ REAL SYNC INTERVAL: {real_sync_interval}")
    print(f"+++ MIN PEERS: {MIN_PEERS}")
    print(f"+++ NODE WEIGHTAGE: {WEIGHTAGE}")
    print(f"+++ BATCH SIZE: {BATCH_SIZE}")
    print(f"+++ MODEL NAME: {BASE_NAME}")
    print(f"+++ TF VERSION: {tf.version.VERSION}")
    print("-------------------")
    
    swarmCallback = SwarmCallback(
        sync_interval=real_sync_interval,
        min_peers=MIN_PEERS,
        node_weightage=WEIGHTAGE,
        val_data=(test_images, test_labels),
        val_batch_size=BATCH_SIZE,
        model_name=BASE_NAME,
        use_adaptive_sync=False
    )
    
    callbacks.append(swarmCallback)


# In[ ]:


# set data augmentation
print('Using real-time data augmentation.')
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.125,
    height_shift_range=0.125,
    fill_mode='constant',
    cval=0,
)

datagen.fit(train_images)
train_flow = datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)


# In[ ]:


history = model.fit(train_flow, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=callbacks)


# In[ ]:


json.dump(history.history, open(f'./history/{NAME}.json', 'w'))

