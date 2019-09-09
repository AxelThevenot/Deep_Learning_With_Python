import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

# region fix error CUBLAS_STATUS_ALLOC_FAILED
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
# endregion

max_features = 10000
max_length = 500
batch_size = 128
output_size = 32

# region load data
print('Loadind data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), ' train sequences')
print(len(input_test), ' test sequences')

print('\n Pad sequences (sample x time)')
input_train = sequence.pad_sequences(input_train, maxlen=max_length)
input_test = sequence.pad_sequences(input_test, maxlen=max_length)

print('input_train shape : ', input_train.shape)
print('input_test shape : ', input_test.shape)
# endregion

# region model and training
model = Sequential()
model.add(Embedding(max_features, output_size))
model.add(SimpleRNN(output_size))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(input_train, y_train,
                    epochs=10, batch_size=batch_size,
                    validation_split=0.3)
# endregion

# region plot
history_dict = history.history
acc, val_acc = history_dict['acc'], history_dict['val_acc']
loss, val_loss = history_dict['loss'], history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.legend()

plt.show()
# endregion