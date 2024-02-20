import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow.keras as K
import datetime

slnum = os.getenv('SLNUM')
min_peers = int(os.getenv('MIN_PEERS', '1'))
group_str = os.getenv('GROUP_STR', '')
ltype = 'SL' if slnum is not None else 'LL'
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model_base_name = f'IMDB-{ltype}-{group_str}'
model_name = f'{model_base_name}-{slnum}-{current_time}'

batch_size = 64
epochs = 10

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

NUM_WORDS=8000

# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=NUM_WORDS)
# np.savez('imdb_data.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

load = np.load('imdb_data.npz')
X_train, y_train = load['X_train'], load['y_train']
X_test, y_test = load['X_test'], load['y_test']

# X_train = K.preprocessing.sequence.pad_sequences(X_train, maxlen=256, padding='post')
# X_test = K.preprocessing.sequence.pad_sequences(X_test, maxlen=256, padding='post')


embed_size = 128
lstm_out = 64
model = K.Sequential([
    K.layers.Embedding(NUM_WORDS, embed_size),
    K.layers.Bidirectional(K.layers.LSTM(lstm_out, return_sequences = True)),
    # K.layers.LSTM(lstm_out),

    K.layers.GlobalMaxPool1D(),
    K.layers.BatchNormalization(),
    K.layers.Dropout(0.05),

    # K.layers.Dense(128, activation="relu", kernel_regularizer=K.regularizers.l2(0.001)),
    # K.layers.BatchNormalization(),
    # K.layers.Dropout(0.05),

    K.layers.Dense(1, activation="sigmoid", kernel_regularizer=K.regularizers.l2(0.001)),
])
model.build()
scheduler = K.optimizers.schedules.ExponentialDecay(0.01, decay_steps=100, decay_rate=0.95)
optimizer = K.optimizers.RMSprop(scheduler)
# optimizer = K.optimizers.Adam()
# optimizer = K.optimizers.SGD(learning_rate=0.001, momentum=0.1)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()



callback = [
    # K.callbacks.EarlyStopping(monitor='val_loss', patience=2),
    K.callbacks.ModelCheckpoint(filepath=f'saved_models/{model_name}.h5', save_best_only=False),
    K.callbacks.TensorBoard(log_dir=f'logs/{model_name}', histogram_freq=1, write_graph=True, write_images=True)
]


if slnum is not None:
    # SYNC_INTERVAL = (len(y_train) + batch_size - 1) // batch_size // 2
    SYNC_INTERVAL = 50
    from swarm import SwarmCallback
    real_sync_interval = SYNC_INTERVAL  # *step_per_epoch

    print("-------------------")
    print("SWARM-LERANING!")
    print(f"+++ REAL SYNC INTERVAL: {real_sync_interval}")
    print(f"+++ MIN PEERS: {min_peers}")
    # print(f"+++ NODE WEIGHTAGE: {NODE_WEIGHTAGE}")
    print(f"+++ BATCH SIZE: {batch_size}")
    print(f"+++ MODEL BASE NAME: {model_base_name}")
    print(f"+++ TF VERSION: {tf.version.VERSION}")
    print("-------------------")

    swarmCallback = SwarmCallback(
        sync_interval=real_sync_interval,
        min_peers=min_peers,
        # node_weightage=NODE_WEIGHTAGE,
        val_data=(X_test, y_test),
        val_batch_size=batch_size,
        model_name=model_base_name,
        use_adaptive_sync=False
    )
    callback.append(swarmCallback)


# K.utils.plot_model(model, to_file='model.png')

model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=callback
)