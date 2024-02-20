import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from datetime import datetime
import os
from tensorflow.keras import callbacks
from tensorflow.python.keras.applications.densenet import DenseNet
from tensorflow.python.util.tf_export import keras_export
import keras_tuner as kt
import argparse
import pickle
import json

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

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


BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 250
if os.path.exists('config.json'):
    with open('config.json') as f:
        config = json.load(f)
        BATCH_SIZE = config['batch_size']
        LEARNING_RATE = config['learning_rate']
        EPOCHS = config['epochs']
        pass
    pass
INPUT_SHAPE = (256, 256)
CLASSES = 14
WEIGHT = None
FREEZE_LAYERS = False
FINAL_DROPOUT = 0.15
EARLY_STOP = None

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE)
args = parser.parse_args()
BATCH_SIZE = args.batch

# MODEL = K.applications.EfficientNetB1
# MODEL=K.applications.DenseNet121
MODEL=DenseNet49

SYNC_INTERVAL = 900
MIN_PEERS = 3
NODE_WEIGHTAGE = int(os.getenv('WEIGHTAGE', '50'))

SLNUM = os.getenv('SLNUM')
SNNUM = os.getenv('SNNUM')
LMODE = os.getenv('LMODE', 'CL')
DATA_DIR = os.getenv('DATA_DIR', './')
TRAIN_VAL_LIST_DIR = os.getenv('TRAIN_VAL_LIST_DIR', './')
SWARM_LEARNING = (SLNUM is not None)

TRAIN_VAL_LIST = os.path.join(TRAIN_VAL_LIST_DIR, 'train_val_list.txt')
DATA_ENTRY_CSV = os.path.join(DATA_DIR, 'NIH_Chest_X-rays/Data_Entry_2017.csv')

# Global Test Set
TEST_LIST = os.path.join(DATA_DIR, 'NIH_Chest_X-rays/test_list.txt')

# Node Specific Test Set
# TEST_LIST=os.path.join(TRAIN_VAL_LIST_DIR, 'test_list.txt')

IMAGES_DIR = os.path.join(DATA_DIR, 'NIH_Chest_X-rays/images')


base_model_name = 'NIHXray-{0}-BS{1}-LR{2}'.format(
    MODEL.__name__, BATCH_SIZE, LEARNING_RATE)
if WEIGHT is not None:
    base_model_name += f'-{WEIGHT}'
else:
    base_model_name += '-random'

if FREEZE_LAYERS:
    base_model_name += '-freeze'
else:
    base_model_name += '-notfreeze'

if SWARM_LEARNING:
    LMODE='SL'
    base_model_name += f'-SL_{SNNUM}_{SLNUM}_W{NODE_WEIGHTAGE}_SINT{SYNC_INTERVAL}'
else:
    base_model_name += f'-{LMODE}'
model_name = base_model_name + \
    datetime.strftime(datetime.now(), '-%Y%m%d%H%M%S')
print(model_name)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    # lr = 1e-3
    lr = LEARNING_RATE
    if epoch > 160:
        lr *= 5e-4
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

df = pd.read_csv(DATA_ENTRY_CSV, index_col='Image Index', converters={'Finding Labels': lambda x: [i for i in x.split('|') if i != 'No Finding']}).drop(
    ['Follow-up #', 'Patient ID', 'Patient Age',
     'Patient Gender', 'View Position', 'OriginalImage[Width', 'Height]',
     'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11'], axis=1
)
df.head()


S = sorted(set(y
               for x in df['Finding Labels']
               for y in x
               ))

all_labels = list(S)


with open(TRAIN_VAL_LIST, 'r') as fp:
    idx = [x.strip() for x in fp.readlines()]
train_df = df.loc[idx].reset_index()
model_name += f'-TRAIN_N{len(train_df)}'
train_df.head()


with open(TEST_LIST, 'r') as fp:
    idx = [x.strip() for x in fp.readlines()]
test_df = df.loc[idx].reset_index()
test_df.head()

if MODEL.__name__.startswith('EfficientNet'):
    datagen = K.preprocessing.image.ImageDataGenerator(
        height_shift_range=5/100,
        width_shift_range=5/100,
        rotation_range=5,
        zoom_range=15/100,
    )
else:
    datagen = K.preprocessing.image.ImageDataGenerator(
        height_shift_range=5/100,
        width_shift_range=5/100,
        rotation_range=5,
        zoom_range=15/100,
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )


train_generator = datagen.flow_from_dataframe(
    train_df,
    IMAGES_DIR,
    x_col='Image Index',
    y_col='Finding Labels',
    target_size=INPUT_SHAPE,
    color_mode='rgb',
    classes=all_labels,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_generator = datagen.flow_from_dataframe(
    test_df,
    IMAGES_DIR,
    x_col='Image Index',
    y_col='Finding Labels',
    target_size=INPUT_SHAPE,
    color_mode='rgb',
    classes=all_labels,
    batch_size=BATCH_SIZE
)

BASE_MODEL = MODEL(
    weights=WEIGHT,
    include_top=False,
    input_shape=train_generator.image_shape,
    pooling='avg'
)
if FREEZE_LAYERS:
    BASE_MODEL.trainable = False


def model_builder(base_model, learning_rate=LEARNING_RATE, final_dropout=FINAL_DROPOUT, classes=CLASSES):
    model = K.Sequential([
        base_model,
        K.layers.Dropout(final_dropout),
        # K.layers.Dropout(FINAL_DROPOUT),
        K.layers.Dense(classes, activation='sigmoid')
    ])
    # ed = K.optimizers.schedules.ExponentialDecay(
    #     learning_rate, decay_step, 0.96)
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=learning_rate),
        loss=K.losses.BinaryCrossentropy(),
        metrics=[
            # , num_labels=CLASSES),
            K.metrics.AUC(name='roc_auc', curve='ROC', multi_label=True),
            # , num_labels=CLASSES),
            K.metrics.AUC(name='pr_auc', curve='PR', multi_label=True),
            K.metrics.BinaryAccuracy(name='accuracy'),
        ]
    )
    return model

def densenet(num_classes, input_shape):
    from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
    from tensorflow.keras.layers import Input, Flatten, Dropout
    from tensorflow.keras.layers import concatenate, Activation
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.models import Model
    # training parameters
    batch_size = 32
    epochs = 200
    data_augmentation = True

    # network parameters
    num_dense_blocks = 3
    use_max_pool = False

    # DenseNet-BC with dataset augmentation
    # Growth rate   | Depth |  Accuracy (paper)| Accuracy (this)      |
    # 12            | 100   |  95.49%          | 93.74%               |
    # 24            | 250   |  96.38%          | requires big mem GPU |
    # 40            | 190   |  96.54%          | requires big mem GPU |
    growth_rate = 12
    depth = 100
    num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
    compression_factor = 0.5

    num_filters_bef_dense_block = 2 * growth_rate

    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(num_filters_bef_dense_block,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal')(x)
    x = concatenate([inputs, x])

    # stack of dense blocks bridged by transition layers
    for i in range(num_dense_blocks):
        # a dense block is a stack of bottleneck layers
        for j in range(num_bottleneck_layers):
            y = BatchNormalization()(x)
            y = Activation('relu')(y)
            y = Conv2D(4 * growth_rate,
                    kernel_size=1,
                    padding='same',
                    kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = Dropout(0.2)(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(growth_rate,
                    kernel_size=3,
                    padding='same',
                    kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = Dropout(0.2)(y)
            x = concatenate([x, y])

        # no transition layer after the last dense block
        if i == num_dense_blocks - 1:
            continue

        # transition layer compresses num of feature maps and reduces the size by 2
        num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
        num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
        y = BatchNormalization()(x)
        y = Conv2D(num_filters_bef_dense_block,
                kernel_size=1,
                padding='same',
                kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = Dropout(0.2)(y)
        x = AveragePooling2D()(y)


    # add classifier on top
    # after average pooling, size of feature map is 1 x 1
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    kernel_initializer='he_normal',
                    activation='softmax')(y)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(1e-3),
                metrics=['acc'])
    return model


def hp_model(hp: kt.HyperParameters):
    model = model_builder(
        BASE_MODEL,
        hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4]),
        0.15,
        CLASSES,
    )
    return model


# base_model = MODEL(**MODEL_ARGS)
# if FREEZE_LAYERS:
#     base_model.trainable = False
#
# model = K.Sequential([
#     base_model,
#     K.layers.Dropout(FINAL_DROPOUT),
#     K.layers.Dense(CLASSES, activation='sigmoid')
# ])
# model.summary()

# class MyTuner(kt.tuners.Hyperband):
#     def run_trial(self, trial, *args, **kwargs):
#         # You can add additional HyperParameters for preprocessing and custom training loops
#         # via overriding `run_trial`
#         kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [8, 16, 32, 64])
#         super(MyTuner, self).run_trial(trial, *args, **kwargs)

def tune():
    MODEL_DIR = 'keras_tuner'
    tuner = kt.Hyperband(
        hp_model,
        objective=kt.Objective('val_pr_auc', direction='max'),
        max_epochs=int(EPOCHS*1.7),
        factor=3,
        directory=MODEL_DIR,
        project_name=f'nih_hp_bs{BATCH_SIZE}',
    )

    callbacks = [
        K.callbacks.EarlyStopping(monitor='val_loss', patience=10),
    ]

    tuner.search(train_generator, epochs=50, validation_data=test_generator, callbacks=callbacks)
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)

    json.dump(best_hps.get_config(), open(f'bs_{BATCH_SIZE}-best_hps.json', 'w'))

def train():
    global model_name

    model = model_builder(BASE_MODEL)
    # model = densenet(14, (256, 256, 3))
    # model_name += 'densenet'

    model_name += f'-{model.optimizer._name}'

    callbacks = [
        K.callbacks.ModelCheckpoint('saved_models/' + model_name + '-{epoch:02d}'),
        K.callbacks.TensorBoard('logs/profile/' + model_name),
        K.callbacks.LearningRateScheduler(schedule=lr_schedule),
        # K.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min', min_lr=0.000005, cooldown=5),
    ]

    step_per_epoch = ((train_generator.n+BATCH_SIZE-1) // BATCH_SIZE)
    if SWARM_LEARNING:
        from swarm import SwarmCallback
        real_sync_interval = SYNC_INTERVAL  # *step_per_epoch

        print("-------------------")
        print("SWARM-LERANING!")
        print(f"+++ REAL SYNC INTERVAL: {real_sync_interval}")
        print(f"+++ MIN PEERS: {MIN_PEERS}")
        print(f"+++ NODE WEIGHTAGE: {NODE_WEIGHTAGE}")
        print(f"+++ BATCH SIZE: {BATCH_SIZE}")
        print(f"+++ MODEL NAME: {base_model_name}")
        print(f"+++ TRAIN GENERATOR N: {train_generator.n}")
        print(f"+++ TEST GENERATOR N: {test_generator.n}")
        print(f"+++ TF VERSION: {tf.version.VERSION}")
        print("-------------------")

        swarmCallback = SwarmCallback(
            sync_interval=real_sync_interval,
            min_peers=MIN_PEERS,
            node_weightage=NODE_WEIGHTAGE,
            val_data=test_generator,
            val_batch_size=BATCH_SIZE,
            model_name=base_model_name,
            use_adaptive_sync=False
        )

        callbacks.append(swarmCallback)
    else:
        if EARLY_STOP:
            callbacks.append(K.callbacks.EarlyStopping(
                monitor='val_loss', patience=EARLY_STOP, verbose=1, mode='min'))

    print(model_name)

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        workers=16,
        max_queue_size=32,
        # steps_per_epoch=step_per_epoch,
        # validation_steps=(test_generator.n+BATCH_SIZE-1) // BATCH_SIZE
    )

    pickle.dump(history.history, open(f'history_{model_name}.pkl', 'wb'))

train()
# tune()
