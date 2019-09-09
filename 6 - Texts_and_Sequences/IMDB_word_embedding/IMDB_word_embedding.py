import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# region fix error CUBLAS_STATUS_ALLOC_FAILED
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
# endregion



# region load data
max_features = 10000
max_length = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# zero-padding to have the same dimensionnality
x_train = preprocessing.sequence.pad_sequences(x_train, max_length)
x_test = preprocessing.sequence.pad_sequences(x_test, max_length)

# endregion

# region model
model = Sequential()

model.add(Embedding(max_features, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
# endregion

# region validation and plot
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


history_dict = history.history
loss, validation_loss = history_dict['loss'], history_dict['val_loss']
accuracy, validation_accuracy = history_dict['acc'], history_dict['val_acc']

epochs = range(1, len(loss) + 1)

fig = plt.figure(1)
ax_loss, ax_accuracy = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
ax_loss.plot(epochs, loss, label='Training loss')
ax_loss.plot(epochs, validation_loss, label='Validation loss')
ax_accuracy.plot(epochs, accuracy, label='Training accuracy')
ax_accuracy.plot(epochs, validation_accuracy, label='Validation accuracy')

ax_loss.xlabel = 'Epoch'
ax_accuracy.xlabel = 'Epoch'
ax_loss.ylabel = 'Loss'
ax_accuracy.ylabel = 'Accuracy'
plt.show()
# endregion


