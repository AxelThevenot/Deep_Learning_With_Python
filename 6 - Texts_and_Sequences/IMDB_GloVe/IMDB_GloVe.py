import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# region processing labels
imdb_dir = 'D:/Axel/Documents/IMDB/aclImdb'
train_dir = imdb_dir + '/train'

texts, labels = [], []

for label_type in ['neg', 'pos']:
    dir_name = train_dir + '/' + label_type
    for file_name in os.listdir(dir_name):
        if file_name[-4:] == '.txt':
            f = open(os.path.join(dir_name, file_name), encoding='utf8')
            texts.append(f.read())
            f.close()
            labels.append(label_type == 'pos')
# endregion

# region tokenizing the text
max_length = 100
training_samples = 2000
validation_samples = 10000
max_words = 10000

tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(texts)
sequences = tok.texts_to_sequences(texts)

word_index = tok.word_index
print('Found %s unique token.' % len(word_index))

# zero padding
data = pad_sequences(sequences, maxlen=max_length)

labels = np.asarray(labels)
print('Shape of data tensor : ', data.shape)
print('Shape of label tensor : ', labels.shape)

# shuffle the data/labels
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data, labels = data[indices], labels[indices]

# set the training and validation datasets
x_train, y_train = data[:training_samples], labels[:training_samples]
x_validation = data[training_samples: training_samples + validation_samples]
y_validation = labels[training_samples: training_samples + validation_samples]
# endregion


# region passing GloVe word_embedding file
glove_dir = 'D:\Axel\Documents\GloVe'

embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),  encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embedding_index))
# endregion

# region prepare GloVe word_embedding matrix
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
# endregion

# region model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_length))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# endregion

# region Load pretrained GloVe into 1st layer (Embedding)
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
# endregion

# region training and evaluation
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=32, epochs=10,
                    validation_data=(x_validation, y_validation))
model.save_weights('pretrained_glove_model.h5')
# endregion


# region plot
history_dict = history.history
acc, loss = history_dict['acc'], history_dict['loss']
val_acc, val_loss = history_dict['val_acc'], history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
# endregion
