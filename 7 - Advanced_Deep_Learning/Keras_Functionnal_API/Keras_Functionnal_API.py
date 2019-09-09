from keras import Input, layers, models
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
# region load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
train_labels = np.array([[1 if val == i else 0 for i in range(10)] for _, val in enumerate(train_labels)])

test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
test_labels = np.array([[1 if val == i else 0 for i in range(10)] for _, val in enumerate(test_labels)])
# endregion

# region model
input_shape = (28 * 28, )

input_layer = Input(shape=input_shape)
dense_1_layer = layers.Dense(32, activation='relu')(input_layer)
dense_2_layer = layers.Dense(32, activation='relu')(dense_1_layer)
ouput_layer = layers.Dense(10, activation='sigmoid')(dense_1_layer)

model = models.Model(input_layer, ouput_layer)
model.summary()
model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
# endregion

# region training
history = model.fit(train_images, train_labels, batch_size=32,verbose=0, epochs=20, validation_split=0.2)

mse_test, acc_test = model.evaluate(test_images, test_labels)
print(mse_test, acc_test)
# endregion


history_dict = history.history
print(history_dict.keys())
acc, val_acc = history_dict['acc'], history_dict['val_acc']
loss, val_loss = history_dict['loss'], history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.scatter(epochs[-1], acc_test, label='Test accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.scatter(epochs[-1], mse_test, label='Test loss')
plt.legend()

plt.show()