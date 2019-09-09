import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Embedding, Bidirectional, LSTM,  Dense
from keras.models import Sequential

# region load data
max_features = 10000
max_length = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# zero-padding
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)
# endregion

# region model bidirectional LSTM
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# endregion

# region training and validation
history = model.fit(x_train, y_train, batch_size=256, epochs=10, validation_split=0.3)
# endregion

# region plot
history_dict = history.history
acc, val_acc = history_dict['acc'], history_dict['val_acc']
loss, val_loss = history_dict['loss'], history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('LSTM : Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('LSTM : Loss')

plt.legend()

plt.show()
# endregion