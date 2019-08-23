"""
This module is the actual implementation of the model definition using Keras.
"""
from keras.models import Sequential
from keras import layers
import keras.backend as K


def build(config):
    K.clear_session()

    # embedding_dim = 50

    model = Sequential()
    # model.add(layers.Embedding( input_dim=config['vocab_size'], 
    #                             output_dim=embedding_dim, 
    #                             input_length=config['maxlen']))

    # model.add(layers.Conv1D(128,5,activation = 'relu'))
    # model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

    