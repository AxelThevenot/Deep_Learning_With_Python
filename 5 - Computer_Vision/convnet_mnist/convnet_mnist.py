from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
import matplotlib.pyplot as plt


# region load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape images and scale their values between 0 and 1
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# reshape labels as a one_hot tensor
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)
# endregion


# region model
model = models.Sequential()

# convolution
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))


# classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# endregion


# region evaluation and plot
num_epoch = 50
history_dict = model.fit(train_images, train_labels, batch_size=512,
                    epochs=num_epoch, validation_data=(test_images, test_labels)).history

accuracy, validation_accuracy = history_dict['acc'], history_dict['val_acc']

epochs = range(1, len(accuracy) + 1)

fig = plt.figure(1)
ax_accuracy = fig.add_subplot(1, 1, 1)
ax_accuracy.plot(epochs, accuracy, label='Training accuracy')
ax_accuracy.plot(epochs, validation_accuracy, label='Validation accuracy')
plt.legend()
plt.show()

# endregion