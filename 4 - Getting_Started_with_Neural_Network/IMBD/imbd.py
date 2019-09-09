import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers

# region dataset
dim = 10000
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)

def vectorize(sequences, dim):
    res = np.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        res[i, seq] = 1
    return res

x_train, x_test = vectorize(train_data, dim), vectorize(test_data, dim)
y_train, y_test = np.asarray(train_label).astype('float32'), np.asarray(test_label).astype('float32')
# endregion


# region model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(dim,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# endregion

# region training & output
model.compile(optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

n_validation = 12500
# Validation samples
x_validation, y_validation = x_train[:n_validation], y_train[:n_validation]
partial_x_validation, partial_y_validation = x_train[n_validation:], y_train[n_validation:]

history = model.fit(x_validation, y_validation,
                     epochs=20,
                     batch_size=512,
                     validation_data=(partial_x_validation, partial_y_validation))

# region plot
import matplotlib.pyplot as plt
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
plt.legend()
plt.show()
# endregion

# endregion








