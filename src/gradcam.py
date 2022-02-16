import keras
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import logging

logger = logging.getLogger ('visualize')
logger.setLevel (logging.INFO)

formatter = logging.Formatter ('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler ('../logs/visualize.log')
file_handler.setFormatter (formatter)

logger.addHandler (file_handler)

class GradCAM:
    def __init__(self, model, layerName, image):
        self.model = model
        self.layerName = layerName
        self.image = image
        logger.info(f'gradcam got model: {model} , layerName: {layerName}')

    
    def get_img_array(self):
        # `array` is a float32 Numpy array of shape (112, 112, 3)
        array = keras.preprocessing.image.img_to_array(self.image)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        logger.info(f'got image array')
        return array


    def make_gradcam_heatmap(self, img_array, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.layerName).output, self.model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            layerName, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, layerName)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        layerName = layerName[0]
        heatmap = layerName @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        logger.info(f'got heatmap')
        return heatmap.numpy()

    def save_and_display_gradcam(self, original_img, heatmap, alpha=0.4):
        # Load the original image
        img = keras.preprocessing.image.img_to_array(original_img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Display Grad CAM
        return(superimposed_img)
