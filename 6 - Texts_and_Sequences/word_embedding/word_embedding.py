import numpy as np
import string
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

def one_hot_encode_word(samples, max_length=10):
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    res = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            res[i, j, index] = 1
    return res

def one_hot_encode_char(samples, max_length=50):
    characters = string.printable
    token_index = dict(zip(range(1, len(characters) + 1), characters))

    res = np.zeros((len(samples),
                    max_length,
                    max(token_index.keys()) + 1))

    for i, sample in enumerate(samples):
        for j, char in enumerate(sample):
            index = token_index.get(char)
            res[i,j, index] = 1
    return res

def one_hot_encode_wKeras(samples, num_words=1000):
    # create tokenizer configured to take account into 1000 most common words
    tok = Tokenizer(num_words=num_words)
    # build the word index
    tok.fit_on_texts(samples)

    # turn strings into list of integers
    sequences = tok.texts_to_sequences(samples)

    # to have also directly the binary representation
    one_hot_res = tok.texts_to_matrix(samples, mode='binary')

    # to recover the word index that was computed
    word_index = tok.word_index

    return one_hot_res

print(one_hot_encode_word(samples))
print(one_hot_encode_char(samples))
print(one_hot_encode_wKeras(samples))