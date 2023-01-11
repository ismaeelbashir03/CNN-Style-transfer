# importing libraries we need
import tensorflow_hub as tf_hub # used to download our pre trained model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# creating a function to load our images
def load_image(path):

    # we load in the image
    img = tf.io.read_file(path)

    # we decode the image
    img = tf.image.decode_image(img, channels = 3)

    # we convert the image to a float 32 bit
    img = tf.image.convert_image_dtype(img, tf.float32)

    # making sure our image is in an array
    img = img[tf.newaxis, :]

    # returning our image
    return img

# reading in our style and content images
style_image = load_image('style-2.jpeg')
content_image = load_image('content-2.jpeg')

# showing our origional images
plt.imshow(np.squeeze(content_image))
plt.show()
plt.imshow(np.squeeze(style_image))
plt.show()

# creating our stylised image
# this model uses Convolutional Neural network layers (feature maps) to first extract
# content from the content image, it does this by using higher level features (ears, nose)
# then the model uses lower level feature maps (edges, pixel colours) dot producted together 
# to get a style and colour, then the model combines the both to get our new stylised image.
# loss is calculated with two weights, one for content one for style, these hyper params can
# be adjusted to train a model to be more stylised or more like the content image
# here we are getting the first result.
stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

# printing this image
plt.imshow(np.squeeze(stylized_image))
plt.show()

# saving our result, np sqeeze in used as the image is in a array
cv2.imwrite('generated_image.jpg', cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))