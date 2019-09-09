import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import backend as K

# region model
model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0
# endregion






# region generate filter visualization
# create a method to transform the inputs image data array to an image (matrix of pixel RGB)
def array_to_image(x):
    x -= x.mean()
    x /= x.std() + 1e-5
    x *= 0.1
    x += 0.5
    x = (np.clip(x, 0, 1) * 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[: ,:, :, filter_index])

    # region obtain the gradient of the loss
    grads = K.gradients(loss, model.input)[0]
    # to normalize
    grads /= (K.sqrt((K.mean(K.square(grads)))) + 1e-5)
    # fetching output values for input values
    iterate = K.function([model.input], [loss, grads])

    # region loss maximization
    # create a gray image with some noise
    input_image_data = np.random.random((1, size, size, 3)) * 20 + 128

    step = 1  # magnitude of each gradient ascent
    epoch = 40  # number of gradient ascent
    for _ in range(epoch):
        # obtain the grads
        loss_value, grads_value = iterate([input_image_data])
        # gradient ascent time
        input_image_data += grads_value * step
    # endregion
    # endregion

    return array_to_image(input_image_data[0])
# endregion

# region generate a grid of filters
def generate_grid_pattern(layer_name, size=64, margin=5):
    sqrt_size = int(size**0.5)
    res_size = sqrt_size * size + (sqrt_size - 1) * margin
    res = np.zeros((res_size, res_size, 3))
    for i in range(sqrt_size):
        for j in range(sqrt_size):
            filter_img = generate_pattern(layer_name, i + (j * sqrt_size), size=size)

            h_start, v_start = i * size + i * margin, j * size + j * margin
            h_end, v_end = h_start + size, v_start + size

            res[h_start:h_end, v_start:v_end, :] = filter_img
    return res.astype('uint8')
# endregion

if __name__ == '__main__':
    plt.imshow(generate_pattern(layer_name, filter_index))
    plt.show()

    plt.figure(figsize=(20, 20))
    plt.imshow(generate_grid_pattern('block1_conv1'))
    plt.show()
