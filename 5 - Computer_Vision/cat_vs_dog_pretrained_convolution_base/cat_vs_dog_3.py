import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.applications import VGG16

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras



# region RUN ONLY ONCE

# change these values if wanted
num_train = 1000
num_validation = 500
num_test = 500
original_dataset_dir = 'D:/Axel/Documents/cat_vs_dog'
original_train_dir = original_dataset_dir + '/train'
original_test_dir = original_dataset_dir + '/train'

# create the directories's path
base_dir = original_dataset_dir + '/cat_vs_dog_smaller'
train_dir = base_dir + '/train'
validation_dir = base_dir + '/validation'
test_dir = base_dir + '/test'

train_cats_dir = train_dir + '/cats'
train_dogs_dir = train_dir + '/dogs'
validation_cats_dir = validation_dir + '/cats'
validation_dogs_dir = validation_dir + '/dogs'
test_cats_dir = test_dir + '/cats'
test_dogs_dir = test_dir + '/dogs'
"""
# create the directories
os.mkdir(base_dir)
os.mkdir(train_dir)
os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)
os.mkdir(validation_dir)
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)
os.mkdir(test_dir)
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

# create a method to copy files from a dir to an other to not repeat the code too many times
def copy_file(cat, from_path, to_path, range):
    pref = 'cat'
    if not cat:
        pref = 'dog'
    file_names = [pref + '.{}.jpg'.format(i) for i in range]
    for f in file_names:
        src = os.path.join(from_path, f)
        dst = os.path.join(to_path, f)
        shutil.copyfile(src, dst)
    print('total from ' + from_path + " to " + to_path + " : " + str(len(os.listdir(to_path))))


range_train = range(num_train)
range_validation = range(num_train, num_train + num_validation)
range_test = range(num_train + num_validation, num_train + num_validation + num_test)

copy_file(True, original_train_dir, train_cats_dir, range_train)
copy_file(True, original_train_dir, validation_cats_dir, range_validation)
copy_file(True, original_test_dir, test_cats_dir, range_test)
copy_file(False, original_train_dir, train_dogs_dir, range_train)
copy_file(False, original_train_dir, validation_dogs_dir, range_validation)
copy_file(False, original_test_dir, test_dogs_dir, range_test)
"""
# endregion

# region model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

# endregion

# region generators and pretrained convolution base

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
datagen = idg(rescale=1./255)
batch_size = 1
def extract_features(directory, num_sample):
    features = np.zeros(shape=(num_sample, 4, 4, 512))
    labels = np.zeros(shape=(num_sample))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= num_sample:
            break
    return np.reshape(features, (num_sample, 4 * 4 * 512)), labels
print("0")
train_features, train_labels = extract_features(train_dir, 2000)
print("1")
validation_features, validation_labels = extract_features(validation_dir, 1000)
print("2")
test_features, test_labels = extract_features(test_dir, 1000)
print("3")
# endregion

# region training
history = model.fit(train_features, train_labels, epochs=30,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))
model.save('cat_vs_dog_small_3.h5')
# endregion

def smooth(points, starting_point=0, factor=0.9):
    smoothed_points = [starting_point]
    for p in points[starting_point + 1:]:
        smoothed_points.append(smoothed_points[-1] * factor + p * (1 - factor))
    return smoothed_points

# region plotting
history_dict = history.history
loss, validation_loss = history_dict['loss'], history_dict['val_loss']
accuracy, validation_accuracy = history_dict['acc'], history_dict['val_acc']

# smooth curves
loss, validation_loss = smooth(loss), smooth(validation_loss)
accuracy, validation_accuracy = smooth(accuracy), smooth(validation_accuracy)

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
