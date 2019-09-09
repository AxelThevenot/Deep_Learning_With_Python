import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import models
from keras import layers

# region import dataset & vectorization & one_hot
num_word = 10000
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_word)

def vectorize(sequences, dim=num_word):
    res = np.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        res[i, seq] = 1
    return res

def one_hot(labels, dim=46):
    res = np.zeros((len(labels), dim))
    for i, lab in enumerate(labels):
        res[i, lab] = 1
    return res

x_train, x_test = vectorize(train_data), vectorize(test_data)
one_hot_train_labels, one_hot_test_labels = one_hot(train_labels), one_hot(test_labels)
# the line below could be done with the method keras.utils.np_utils.to_categorical(labels)
# endregion

# region model
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(num_word, )))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# endregion

# region validation
num_val = 1000
x_val = x_train[:num_val]
partial_x_val = x_train[num_val:]

y_val = one_hot_train_labels[:num_val]
partial_y_val = one_hot_train_labels[num_val:]

history = model.fit(partial_x_val, partial_y_val, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# endregion

# region plotting result
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
plt.show()

# endregion

