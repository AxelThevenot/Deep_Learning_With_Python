import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, concatenate, Dense
from keras.optimizers import RMSprop



text_vocabulary_size = 500
question_vocabulary_size = 300
answer_vocabulary_size = 3

# region load data (this is false data !)
n_samples = 500
max_length = 100
ratio_train = 2 / 3
ratio_validation = 0.25

# Non-sense data indeed...
texts = np.array([[int(3 * i / (j + 1) + j + np.random.randint(0, 1 + (i % (j + 1))))
                   for j in range(max_length)]
                    for i in range(n_samples)])
questions = np.array([[int(j / (i + 1) + i**2 / (j + 1) + np.random.randint(0, 1 + (i % (j + 1))))
                    for j in range(max_length)]
                     for i in range(n_samples)])
answers = np.array([[int(i + j)
                    for j in range(answer_vocabulary_size)]
                     for i in range(n_samples)])


# shuffle the data
indices = np.arange(n_samples)
np.random.shuffle(indices)
texts[:], questions[:], answers[:] = texts[indices], questions[indices], answers[indices]

# split into training and testing set
n_split = int(n_samples * ratio_train)
x_train = [texts[n_split:], questions[n_split:]]
x_test = [texts[:n_split + 1], questions[:n_split + 1]]

y_train = answers[n_split:]
y_test = answers[:n_split + 1]
# endregion

# region model
# region 1st branch
text_input_layer = Input(shape=(None,), dtype='float', name='text')
text_embedded_layer = Embedding(16, text_vocabulary_size)(text_input_layer)
text_encoded_layer = LSTM(16)(text_embedded_layer)
# endregion

# region 2nd branch
question_input_layer = Input(shape=(None,), dtype='int32', name='question')
question_embedded_layer = Embedding(16, question_vocabulary_size)(question_input_layer)
question_encoded_layer = LSTM(16)(question_embedded_layer)
# endregion

# region concatenation
concatenated_layer = concatenate([text_encoded_layer, question_encoded_layer], axis=-1)
answer_layer = Dense(answer_vocabulary_size, activation='softmax')(concatenated_layer)
# endregion

model = Model([text_input_layer, question_input_layer], answer_layer)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
# endregion


# region training and plot
history = model.fit(x_train, y_train, verbose=1, batch_size=1, epochs=10, validation_split=ratio_validation)

loss_test, acc_test = model.evaluate(x_test, y_test)
print(loss_test, acc_test)

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
plt.scatter(epochs[-1], loss_test, label='Test loss')
plt.legend()

plt.show()
# endregion
