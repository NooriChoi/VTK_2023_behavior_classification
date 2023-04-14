# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:33:57 2023

@author: Noori
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Masking

def train(x_train, learning_rate, batch_size, epochs):
    # define model
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(100, activation='tanh'))
    model.add(RepeatVector(x_train.shape[1]))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(x_train.shape[2])))
    model.summary()
    
    # compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    # train_model
    model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=True)
    
    # separate encoder & decoder
    ## encoder
    Encoder = Model(inputs=model.inputs, outputs=model.layers[1].output)
    ## decoder
    encoded_input = tf.keras.Input(shape=(100,))
    decoded = model.layers[2](encoded_input)
    decoded = model.layers[3](decoded)
    decoded = model.layers[4](decoded)
    Decoder = Model(encoded_input, decoded)
    
    return model, Encoder, Decoder