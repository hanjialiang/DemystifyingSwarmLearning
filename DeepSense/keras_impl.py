#!/usr/bin/env python
# coding: utf-8

"""
export SLNUM=2
export SNNUM=1
docker container rm sl-$SLNUM
bash ./swarm-learning/bin/run-sl        \
    --name=sl-$SLNUM                         \
    --network sl-net             \
    --sl-platform=TF                   \
    --host-ip=sl-$SLNUM                \
    --sn-ip=sn-$SNNUM                   \
    --data-dir=/home/yudonghan/SwarmSense/DeepSense/data  \
    --model-dir=/home/yudonghan/SwarmSense/DeepSense/model/$SLNUM    \
    --model-program=keras_impl.py        \
    --apls-ip apls                 \
    --gpu=0                      \
    -serverAddress spire-server            \
    -genJoinToken
"""

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import tensorflow.keras as keras
from os.path import join as ospj
import os
from swarm import SwarmCallback

print("LISTDIR:::::",os.listdir('.'))
print("ENV:::::::::",os.getenv('DATA_DIR', './data'))
print("CUDA_VISIBLE:::::::::",os.getenv('CUDA_VISIBLE_DEVICES'))
print("DEV:::::::::",os.listdir('/dev'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048+1024+512)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# In[2]:

DATA_DIR=os.getenv('DATA_DIR', './data')
DATA_ROOT=ospj(DATA_DIR, 'ruifeng')
BATCH_SIZE=32


# In[3]:


data = None
with open(ospj(DATA_DIR, 'ruifeng_by_date.json'), 'r') as fp:
    data = json.load(fp)
print(data.keys())


# In[4]:


print(data['img_size'])
print(data['label_index'])


# In[5]:


img=plt.imread(ospj(DATA_ROOT, data['train']['pos'][0][0]))
print(img.shape)
plt.imshow(img)


# In[6]:


def get_dataset(pos_list, neg_list, data_root='data', training=True, batch_size=BATCH_SIZE):
    def img_pre_proc(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, data['img_size'][:2])
        image = tf.cast(image, tf.float32)
        image = keras.applications.densenet.preprocess_input(image)
        # image = tf.image.per_image_standardization(image)
        return image

    def label_pre_proc(label):
        idx = 0 if label == data['label_index']['Ok'] else 1
        return tf.one_hot(idx, depth=2)
#        return 0 if label == data['label_index']['Ok'] else 1
    
    print('Total pos count:', len(pos_list))
    print('Total neg count:', len(neg_list))
    lst = pos_list + neg_list
    #print(len(lst), len(lst)%batch_size)
    img_lst, label_lst = zip(*lst)
    img_lst = [ospj(data_root, x) for x in img_lst]
    
    img_ds = tf.data.Dataset.from_tensor_slices(img_lst)
    img_ds = img_ds.map(img_pre_proc)

    label_ds = tf.data.Dataset.from_tensor_slices(list(label_lst))
    label_ds = label_ds.map(label_pre_proc)

    ds = tf.data.Dataset.zip((img_ds, label_ds))
    if training:
        ds = ds.shuffle(buffer_size=len(lst))
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


# In[7]:


train_dataset = get_dataset(data_root=DATA_ROOT, pos_list=data['train']['pos'], neg_list=data['train']['neg'], training=True)
val_dataset = get_dataset(data_root=DATA_ROOT, pos_list=data['val']['pos'], neg_list=data['val']['neg'], training=False)
steps_per_epoch=np.math.ceil((len(data['train']['pos'])+len(data['train']['neg']))/BATCH_SIZE)
steps_per_epoch


# In[8]:
#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
from tensorflow.python.keras.applications.densenet import DenseNet
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.applications.densenet.DenseNet169',
              'keras.applications.DenseNet169')
def DenseNet49(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Densenet169 architecture."""
    return DenseNet([4, 4, 8, 6], include_top, weights, input_tensor, input_shape, pooling, classes)

model = DenseNet49(include_top=True, weights=None, input_shape=data['img_size'], classes=2)
# model = keras.applications.DenseNet121(include_top=True, weights=None, classes=10, input_shape=data['img_size'])

max_epochs = 20
swSyncInterval = steps_per_epoch
min_peers = 2
model_name = 'EfficientNetB2-DeepSense'
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        *[tf.keras.metrics.Recall(name='{0}-Recall'.format(x), class_id=x) for x in range(2)],
        *[tf.keras.metrics.Precision(name='{0}-Precision'.format(x), class_id=x) for x in range(2)]]
)


# In[10]:

swarmCallback = SwarmCallback(
    sync_interval=swSyncInterval,
    min_peers=min_peers,
    val_data=val_dataset,
    val_batch_size=BATCH_SIZE,
    model_name=model_name,
    use_adaptive_sync=False
)

model.fit(
    train_dataset,
    epochs=max_epochs,
    validation_data=val_dataset,
    validation_freq=1,
    steps_per_epoch=steps_per_epoch,
    callbacks=[swarmCallback]
)

model.save('enetv2.h5')


# In[ ]:



