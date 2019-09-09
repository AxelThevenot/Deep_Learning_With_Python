import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras import models

# region fix bugs on GPU
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
# endregion

# region image
img_path = 'D:/Axel/Documents/cat_vs_dog/cat_vs_dog_smaller/train/cats/cat.3.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
# endregion

# region model
model = load_model('cat_vs_dog_small_2.h5')
model.summary()

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
# endregion



# region plot image
def plot_image():
    plt.imshow(img_tensor[0])
    plt.show()
# endregion

# region plot the nth channel
def plot_n_th_channel(n=7):
    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation[0, :, :, n], cmap='viridis')
    plt.show()
# ednregion


# region plot every channel
def plot_every_channel(images_per_row=16):
    layer_names = [layer.name for layer in model.layers[:8]]

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, size * images_per_row))
        scale = 1./size

        for col in range(n_cols):
            for row in range(images_per_row):

                channel_image = layer_activation[0, :, :, col * images_per_row + row]

                # recenter the value to be visual
                channel_image -= channel_image.mean()
                if channel_image.std() != 0:
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128

                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

            plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
# endregion


if __name__ == '__main__':
    plot_image()
    plot_n_th_channel()
    plot_every_channel()
