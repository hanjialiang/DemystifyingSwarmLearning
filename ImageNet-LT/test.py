#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.keras as K
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from datetime import datetime


# In[ ]:


for name in ['HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy']:
    os.environ[name] = 'http://162.105.175.156:3434'


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


file_paths = ['data/ImageNet_LT_test.txt', 'data/ImageNet_LT_val.txt', 'data/ImageNet_LT_train.txt']


# In[ ]:


# df = pd.read_csv(file_paths[2], dtype={1:str}, sep=' ', header=None, converters={0:lambda x:'data/'+x})
# df.head()


# In[ ]:


# img = plt.imread(df[0][333])
# plt.imshow(img)


# In[ ]:


def read_image_and_cast_label(x):
    path = x[0]
    label = x[1]
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = K.applications.efficientnet.preprocess_input(image)
    # image = tf.image.per_image_standardization(image)
    label = tf.strings.to_number(label, out_type=tf.int32)
    one_hot = tf.one_hot(label, depth=1000)
    return image, one_hot


# In[ ]:


def process_dataset(dataset, shuffle_buffer, training=False, batch_size=32):
    ds = dataset.map(read_image_and_cast_label)
    if training:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


# In[ ]:


def read_imagenet_dataset(path, training=False, batch_size=32):
    df = pd.read_csv(path, dtype={1:str}, sep=' ', header=None, converters={0:lambda x:'data/'+x})
    dataset = tf.data.Dataset.from_tensor_slices(df)
    dataset = process_dataset(dataset, len(df), training, batch_size)
    return dataset, len(df)


# In[ ]:


max_epochs=300
batch_size=32

model_name = "EfficientNetB0-Imagenet_LT"
current_name = model_name


# In[ ]:


train_ds, train_tot = read_imagenet_dataset(file_paths[2], True, batch_size)
val_ds, val_tot = read_imagenet_dataset(file_paths[1], False, batch_size)
steps_per_epoch = (train_tot+batch_size-1)//batch_size

# Swarm learning config
SWARM_LEARNING=False
swSyncInterval = steps_per_epoch * 5
min_peers = 3

if SWARM_LEARNING:
    current_name += '-SL{0}-interval{1}-peers{2}'.format(os.getenv('SLNUM'), swSyncInterval, min_peers)
else:
    current_name += '-baseline'
    
current_name += datetime.now().strftime("-%Y%m%d%H%M%S")
print(current_name)


# In[ ]:


model = K.applications.EfficientNetB0(weights=None)


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# model.evaluate(dataset)


# In[ ]:


callbacks=[
    K.callbacks.TensorBoard(
        log_dir="logs/profile/" + current_name,
        profile_batch=5
    ),
    K.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    K.callbacks.ModelCheckpoint('saved_models/' + current_name, save_best_only=True)
]


# In[ ]:


model.fit(
    train_ds,
    epochs=max_epochs,
    validation_data=val_ds,
    validation_freq=1,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks
)

