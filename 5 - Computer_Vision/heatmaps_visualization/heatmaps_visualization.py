import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image


# region load model and image
model = VGG16(weights='imagenet')

img_path = 'creative_commons_elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)  # color normalization
# endregion

# region run and print prediction
predictions = model.predict(x)

print(decode_predictions(predictions, top=3)[0])
n_elephant = np.argmax(predictions[0])  # == 386
# endregion

# region Grad-CAM algorithm
african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

# endregion


# region plot heatmap
# scale the values between 0 and 1
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

plt.matshow(heatmap)
plt.show()
# endregion

# region superimpose heatmap on the original picture
img = cv2.imread(img_path)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img

cv2.imwrite('super_imposed_img.jpg', superimposed_img)
# endregion