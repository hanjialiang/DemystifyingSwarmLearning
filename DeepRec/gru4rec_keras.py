import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow.keras as K

def GRU4REC(input_size, hidden_size, lr=.001, dropout_input=.0, dropout_hidden=.25): 

    inputs = K.layers.Input(shape=(None, 1, input_size))
    #mask = K.layers.Masking()
    gru = K.layers.GRU(
        hidden_size,
        stateful=True,
        return_state=False,
        name='GRU',
        dropout=dropout_input,
        recurrent_dropout=dropout_hidden
    )(inputs=inputs)
    predictions = K.layers.Dense(input_size, activation='softmax')(gru)
    model = K.Model(inputs=inputs, outputs=[predictions], name='GRU4REC')
    model.compile(loss=K.losses.CategoricalCrossentropy(), optimizer=K.optimizers.Adam(lr=lr))
    model.summary()

    return model

def main():
    GRU4REC()