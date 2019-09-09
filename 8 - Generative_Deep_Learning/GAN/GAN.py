import keras
from keras import layers
import numpy as np
import os
from keras.datasets import mnist
from keras.preprocessing import image

# region GAN generator network
latent_dim = 32
height = 28
width = 28
channels = 1

channel_feature_map = 18
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

# region load data (here we will try to generate one digit)
digit_to_generate = 5
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train[np.argwhere(y_train == digit_to_generate)][:, 0]
# keep only the digit to generate

print('{0} {1}-digit founded'.format(x_train.shape[0], digit_to_generate))
x_train = x_train.reshape((x_train.shape[0], ) + (height, width, channels)).astype('float32') / 255
# endregion

# region GAN training
iterations = 100000
batch_size = 20
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
    labels += 1e-5 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start = (start + batch_size) % (len(x_train) - batch_size)
    if step % 100 == 0:
        print('discriminator loss : ', d_loss)
        print('adversarial loss : ', a_loss)
    # save and give information
    if step % 1000 == 0:
        gan.save_weights('gan.h5')

        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_{0}_at_step_{1}.png'.format(digit_to_generate, step)))

        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_{0}_at_step_{1}.png'.format(digit_to_generate, step)))

# endregion

