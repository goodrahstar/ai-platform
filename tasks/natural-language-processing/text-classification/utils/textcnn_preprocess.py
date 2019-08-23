"""
This module is responsible for defining the preprocessing steps on the text data. 
These steps are very speficic to the text cnn model as the name suggest.
"""

import traceback
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
 

def _load_data(config):
    try:
        dataset = pd.read_csv(config['file_path'])
    except:
        traceback.print_exc()
    return dataset


def pre_processing(config):
    
    dataset = _load_data(config)
    sentences = dataset['sentence'].values
    y = dataset['label'].values


    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)


    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1

     
    X_train = pad_sequences(X_train, padding='post', maxlen=config['maxlen'])
    X_test = pad_sequences(X_test, padding='post', maxlen=config['maxlen'])

    return X_train, X_test, y_train, y_test ,vocab_size