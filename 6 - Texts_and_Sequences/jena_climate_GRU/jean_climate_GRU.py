import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, GRU, Dense
from keras.optimizers import RMSprop



data_dir = 'D:/Axel/Documents/Jena_climate'
file_name = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
N_PTS_PER_DAY = 144




# region loading and inspection
f = open(file_name)
data = f.read()
f.close()

lines = data.split('\n')
header, lines = lines[0].split(','), lines[1:]
print(header)
# print(lines)
# endregion


# region parsing the data
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    # remove the date time
    values = [float(x) for x in line.split(',')[1:]]

    float_data[i, :] = values
# endregion

# region plot the temperature timserie
n_day_to_display = 10

temp = float_data[:, 1]  # 2th row is the Temperature T in degC
plt.plot(temp, c='#2F9599')
plt.title('Timeserie of T (degC) from 2009 to 2019')

plt.figure()

plt.plot(temp[:n_day_to_display * N_PTS_PER_DAY], c='#2F9599')
plt.title('Timeserie of T (degC) for the firsts %s day from 2009' % n_day_to_display)

# endregion

# region preparing the data

lookback = 10 * N_PTS_PER_DAY  # lookback 5 days
step = 6  # sample on per hour
delay = 1 * N_PTS_PER_DAY  # target 1 day after
batch_size = 128

train_steps = 200000
val_steps = 100000

# normalize the data
mean, std = float_data[:train_steps].mean(axis=0), float_data[:train_steps].std(axis=0)
float_data = (float_data - mean) / std

# create a mehtod to generate samples and targets
def generator(data, lookback, delay, min_index, max_index, shuffle, batch_size, step):
    """
    generate samples and targets
    :param data: original array of points
    :param lookback: how many timesteps back to the input data should go
    :param delay: how many timesteps in the future the target should go
    :param min_index: delimit which timesteps to draw from
    :param max_index: delimit which timesteps to draw to
    :param shuffle: Whether to shuffle or keep the chronology
    :param batch_size: number of sample per batch
    :param step: period to sample the data
    :return: samples, targets
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    max_index = min(max_index, len(data) - delay - 1)
    i = min_index
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step + 1, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, _ in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j] + 1, step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets

train_gen = generator(float_data, lookback, delay,
                              0,
                              train_steps,
                              True, batch_size, step)
val_gen = generator(float_data, lookback, delay,
                              train_steps + 1,
                              train_steps + val_steps,
                              True, batch_size, step)
test_gen = generator(float_data, lookback, delay,
                              train_steps + val_steps + 1,
                              None,
                              True, batch_size, step)

val_steps = val_steps - 1 - lookback
test_steps = len(float_data) - (train_steps + val_steps + 1) - lookback

print(next(train_gen)[0].shape)
print(next(val_gen)[0].shape)

# endregion

# region model
model = Sequential()


# add layers
model.add(GRU(32, activation='relu', input_shape=(lookback // step + 1, float_data.shape[-1])))
model.add(Dense(1))

# compile the model
model.compile(loss='mse', optimizer=RMSprop())
# endregion

# region training and validation
EPOCH = 20
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=EPOCH,
                              validation_data=val_gen,
                              validation_steps=250)
# endregion


# region plot
history_dict = history.history
loss, val_loss = history_dict['loss'], history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, c='#2F9599', label='Training loss')
plt.plot(epochs, val_loss, c='#8800FF', label='Validation loss')
plt.legend()
plt.title('With GRU')
plt.show()
# endregion

