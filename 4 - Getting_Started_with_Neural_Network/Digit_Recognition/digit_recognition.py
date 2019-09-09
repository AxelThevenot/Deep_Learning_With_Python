from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
train_labels = np.array([[1 if val == i else 0 for i in range(10)] for _, val in enumerate(train_labels)])

test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
test_labels = np.array([[1 if val == i else 0 for i in range(10)] for _, val in enumerate(test_labels)])

model = models.Sequential()
model.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.01),loss='mse',metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=128, epochs=10)
print(model.evaluate(test_images, test_labels, batch_size=128))

