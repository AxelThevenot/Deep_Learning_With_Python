import keras
from keras.models import Sequential
from keras.layers import Conv1D, Embedding, MaxPooling1D, GlobalMaxPooling1D, Dense, Flatten
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np


# region load data
max_features = 2000
max_length = 300

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)
# endregion

# region model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_length, name='embed'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# endregion


# region TensorBoard
dir_path = 'my_log_dir'
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=dir_path,
        histogram_freq=1,
        embeddings_freq=1,
        embeddings_data=np.arange(0, max_length).reshape((1, max_length))
    )
]
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks)
# endregion

