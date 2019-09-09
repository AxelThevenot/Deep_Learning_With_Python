import scipy
import numpy as np
from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image

# region load model
K.set_learning_phase(0)  # unable training

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
model.summary()
# endregion


# region set configuration
layer_contribution = {
    'mixed2': 1.,
    'mixed5': 1.5,
    'mixed7': 2.,
    'mixed9': 3.,
}
# endregion


# region defining the loss to be maximized
layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)

for layer_name in layer_contribution:
    coeff = layer_contribution[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling
# endregion


# region Gradient-ascent process
dream = model.input

# compute the gradient of the dream with regard to the loss
grads = K.gradients(loss, dream)[0]

# normalize the gradient
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# set up a Keras function to retrieve the loss and the gradient given an image
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grads_values = fetch_loss_and_grads([x])
        if max_loss is not None and loss_value > max_loss:
            break
        print('Loss value at iteration {0}/{1} : {2}'.format(i + 1, iterations, loss_value))
        x += step * grads_values
    return x
# endregion

# region auxiliary methods
def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, file_name):
    pil_img = deprocess_img(np.copy(img))
    scipy.misc.imsave(file_name, pil_img)

def preprocess_img(img_path):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_img(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += .5
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# endregion


# region running gradient ascent over different successive scales
step = 0.05  # as a learning rate
num_octave = 3
octave_scale = 1.4
iterations = 20

max_loss = 20.

base_image_path = 'adrien.jpg'

img = preprocess_img(base_image_path)

# prepare the different shapes
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i  in range(1, num_octave):
    next_shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(next_shape)


# reverse the array as it is increasing
successive_shapes = successive_shapes[::-1]

# resize the original image to the smallest shape
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

# scale up the dream image
for shape in successive_shapes:
    print('preprocessing image shape : ', shape)
    img = resize_img(img, shape)
    # runs the gradient ascent on the image
    img = gradient_ascent(img, iterations, step, max_loss)

    # scales up
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_shape_original = resize_img(original_img, shape)

    # retrieve the lost details
    lost_detail = same_shape_original - upscaled_shrunk_original_img

    # reinjects lost details in the subscaled dreams
    img += lost_detail

    shrunk_original_img = resize_img(original_img, shape)

    # save the subscaled dreams
    save_img(img, 'dream_at_scale_' + str(shape) + '.png')

save_img(img, 'final_dream.png')
# endregion

