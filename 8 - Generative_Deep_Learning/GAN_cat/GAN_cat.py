import keras
from keras import layers
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.preprocessing import image


from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras




# region GAN generator network
latent_dim = 32
height = 50
width = 50
channels = 3
original_dataset_dir = 'D:/Axel/Documents/cat_vs_dog'
base_dir = original_dataset_dir + '/cat_vs_dog_smaller'
train_dir = base_dir + '/train'

channel_feature_map = 64
generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(channel_feature_map * (width // 2) * (height // 2))(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((width // 2, height // 2, channel_feature_map))(x)

x = layers.Conv2D(channel_feature_map * 2, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# upsample to width * height
x = layers.Conv2DTranspose(channel_feature_map * 2, 4,strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)


x = layers.Conv2D(channel_feature_map * 2, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channel_feature_map * 2, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# create an image of shape (width, height, channels)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

# endregion

# region GAN discriminator network
conv2d_units = 128

discriminator_input = keras.Input((width, height, channels))

x = layers.Conv2D(conv2d_units, 3)(discriminator_input)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(conv2d_units, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(conv2d_units, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(conv2d_units, 4, strides=2)(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()


discriminator_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=1.)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# endregion



# region adversarial network
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=2e-4, clipvalue=1.)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
gan.summary()
# endregion



# region load data
datagen = idg()
def extract_features(directory, num_sample):
    batch_size = 1
    features = np.zeros(shape=(num_sample, height, height, channels))
    labels = np.zeros(shape=(num_sample,))
    gen = datagen.flow_from_directory(directory,
                                            target_size=(height, width),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in gen:
        features[i * batch_size: (i + 1) * batch_size] = inputs_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= num_sample:
            break
    return np.reshape(features, (num_sample, height * width * channels)), labels

x_train, y_train = extract_features(train_dir, 2000)
print(y_train)
x_train = x_train[np.argwhere(y_train == 1)][:, 0]

# endregion







print('{0} cat founded'.format(x_train.shape[0]))
x_train = x_train.reshape((x_train.shape[0], ) + (height, width, channels)).astype('float32') / 255
# endregion

# region GAN training
iterations = 100000
batch_size = 10
save_dir = 'save'

start = 0
for step in range(iterations + 1):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

    # add noise
    labels += 0.02 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start = (start + batch_size) % (len(x_train) - batch_size)

    # save and give information
    if step % 100 == 0:
        gan.save_weights('gan.h5')
        print('discriminator loss : ', d_loss)
        print('adversarial loss : ', a_loss)
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_cat_at_step_{0}.png'.format(step)))

        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_cat_at_step_{0}.png'.format(step)))

# endregion

