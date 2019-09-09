import keras
import numpy as np
import random
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop

# region load data
aws_origin = 'http://s3.amazonaws.com/text-datasets/nietzsche.txt'
file_name = 'nietzsche.txt'
path = keras.utils.get_file(file_name, origin=aws_origin)

text = open(path).read().lower()
print('Corpus length : ', len(text))
# endregion


# region extract data
max_length = 60
step = 3
sentences, next_char = [], []

for i in range(0, len(text) - max_length, step):
    sentences.append(text[i:i + max_length])
    next_char.append(text[i + max_length])
print('Number of sequences : ', len(sentences))

chars = sorted(list(set(text)))
print('Unique characters : ', len(chars))

char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectoriszation...')

# one_hot encode the characters into binary arrays

x = np.zeros((len(sentences), max_length, len(chars)), dtype='bool')
y = np.zeros((len(sentences), len(chars)), dtype='bool')
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = True
    y[i, char_indices[next_char[i]]] = True

# endregion


# region model
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(chars))))
model.add(Dense(len((chars)), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-2), metrics=['acc'])
# endregion

# region training
batch_size = 128
epochs = 30

# endregion

# region text generation
range_temperature = [0.2, 0.5, 1., 1.2]
n_char_from_seed = 400

def sample(preds, T=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.exp(np.log(preds) / T)
    preds /= np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

for e in range(epochs):
    print('Epoch {0}/{1}'.format(e, epochs))
    model.fit(x, y, batch_size=batch_size, epochs=1)
    start_index = random.randint(0, len(text) - max_length - 1)
    generated_text = text[start_index:start_index + max_length]
    print('--- Generating with seed : "{0}"'.format(generated_text))

    for temperature in range_temperature:
        print('------ temperature : ', temperature)
        sys.stdout.write(generated_text)

        for i in range(n_char_from_seed):
            # one_hot encode the characted generated so far
            sampled = np.zeros((1, max_length, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1

            # sample the next character
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text = generated_text[1:] + next_char
            sys.stdout.write(next_char)

# endregion
