import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import models
from keras import layers

# region load dataset & normalize
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

def normalize(array):
    return (array - np.mean(array, axis=0))/ np.std(array, axis=0)

train_data, test_data = normalize(train_data), normalize(test_data)
# endregion




# region model
def build_model():
    m = models.Sequential()
    m.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    m.add(layers.Dense(64, activation="relu"))
    m.add(layers.Dense(1))
    m.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return m
# endregion

# region validation
K = 4

num_val_samples = len(train_data) // K
num_epoch = 1500
scores = []
mae_histories = []

for i in range(K):
    print('processing fold #', i+1, '/', K)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, batch_size=64,
                        epochs=num_epoch, verbose=0,
                        validation_data=(val_data, val_targets))
    mae_histories.append(history.history['val_mean_absolute_error'])
    """
    # unused next but can bu useful
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    scores.append([val_mse, val_mae])
    """
# endregion

# region plot
average_mae_history = [np.mean([x[i] for x in mae_histories]) for i in range(num_epoch)]

"""
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epoch')
plt.ylabel('Validation MAE')
plt.show()
"""

# as there is a huge variance and a different scale with the first 30 points it is pretty hard to see the graph
# so here we have a method that smooths the curve with a exponential mean and remove the firsts points


def smooth(points, starting_point=30, factor=0.75):
    smoothed_points = [starting_point]
    for p in points[starting_point + 1:]:
        smoothed_points.append(smoothed_points[-1] * factor + p * (1 - factor))
    return np.array(smoothed_points)


smooth_mae_history = smooth(average_mae_history)
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epoch')
plt.ylabel('Validation MAE')
plt.show()

# endregion