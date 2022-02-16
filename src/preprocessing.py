"""
preprocessing module
"""
import numpy as np
from PIL import Image, ImageOps
from skimage.transform import resize
from skimage.color import rgb2gray
import logging

logger = logging.getLogger ('preprocess')
logger.setLevel (logging.INFO)

formatter = logging.Formatter ('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler ('../logs/preprocess.log')
file_handler.setFormatter (formatter)

logger.addHandler (file_handler)


reference_shape = (112, 112, 1)


def preprocess_image(image):
    """
    :param image: mnist image to be processed
    :return: preprocessed image ready for prediction
    """
    # pil preprocessing
    image = np.array(image)
    image = resize(image , (112 ,112))
    image = rgb2gray(image)
    logger.info ('The image is grayscale')
    image = np.reshape(image,(1, 112, 112, 1))
    image = image/255.
    logger.info ('Preprocess done')
    return image

