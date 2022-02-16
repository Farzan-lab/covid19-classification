"""Models are to be implemented here, or if too long in separate files
The obligation that this creates is that you have to include the methods from the base class in your model.
"""
from base_classes import ClassificationModel
from typing import Tuple
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D, Conv1D, Flatten
import logging

logger = logging.getLogger ('models')
logger.setLevel (logging.INFO)

formatter = logging.Formatter ('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler ('../logs/models.log')
file_handler.setFormatter (formatter)

logger.addHandler (file_handler)

class MobileNet(ClassificationModel):
    """
    The mobileNetV2 model
    """
    def __init__(self, model_path: str,
                 input_shape: Tuple[int, int, int] = (112, 112, 1),
                 num_classes: int = 4,
                 pre_trained: bool = True):
        """
        :param model_path: where the model is located
        :param input_shape: input shape for the model to be built with
        :param num_classes: number of classes in the classification problem
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.model = self.__load_model()
        logger.info(f'model: mobileNet , input_shape: {input_shape}, num_classes: {num_classes}, pre_trained: {pre_trained}')

    def __load_model(self):
        """ model loader """
        mobile = MobileNetV2(input_shape= (112,112,3),  include_top=False, pooling='max')
        mobile.trainable = False
        model = Sequential()

        model.add(Conv1D(3, 1, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(mobile)
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='sigmoid'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        
        if self.pre_trained:
            model.load_weights(self.model_path)

        logger.info('mobileNetV2 is ready')
        return model

    def predict(self, images):
        """
        Use the loaded model to make an estimation.
        :param images: photo to make the prediction on.
        :return: predicted class.
        """
        prediction = self.model.predict(images)
        logger.info(f'prediction of mobileNet is: {prediction}')
        return prediction

    
    """----- print summary -----"""
    def summary(self):
        return self.model.summary()

    """----- model fitting for training -----"""
    def fit(self, X, y=None, validation_data: Tuple = None, batch_size=None, epochs=1):
        res = self.model.fit(X, y, 
                            validation_data=validation_data, 
                            batch_size=batch_size, 
                            epochs=epochs)
        return res

    """----- save model as h5 or hdf5 -----"""
    def save_model(self, saved_model_name:str, weights:bool=False):
        if weights:
            self.model.save_weights("../weights/" + saved_model_name + ".hdf5")
        else:
            self.model.save("../weights/" + saved_model_name + ".h5")



class EfficientNet(ClassificationModel):
    """
    The efficientNetB0 model
    """
    def __init__(self, model_path: str,
                 input_shape: Tuple[int, int, int] = (112, 112, 1),
                 num_classes: int = 4,
                 pre_trained: bool = True):
        """
        :param model_path: where the model is located
        :param input_shape: input shape for the model to be built with
        :param num_classes: number of classes in the classification problem
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.model = self.__load_model()
        logger.info(f'model: mobileNet , input_shape: {input_shape}, num_classes: {num_classes}, pre_trained: {pre_trained}')

    def __load_model(self):
        """ model loader """
        efficient_net = EfficientNetB0(input_shape=(112,112,3), include_top=False, pooling='max')
        efficient_net.trainable = False

        model = Sequential()

        model.add(Conv1D(3, 1, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(efficient_net)
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='sigmoid'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        
        if self.pre_trained:
            model.load_weights(self.model_path)
        logger.info('efficientNetB0 is ready')
        return model

    def predict(self, images):
        """
        Use the loaded model to make an estimation.
        :param images: photo to make the prediction on.
        :return: predicted class.
        """
        prediction = self.model.predict(images)
        logger.info(f'prediction of efficientNet is: {prediction}')
        return prediction


    """----- print summary -----"""
    def summary(self):
        return self.model.summary()

    """----- model fitting for training -----"""
    def fit(self, X, y=None, validation_data: Tuple = None, batch_size=None, epochs=1):
        res = self.model.fit(X, y, 
                            validation_data=validation_data, 
                            batch_size=batch_size, 
                            epochs=epochs)
        return res

    """----- save model as h5 or hdf5 -----"""
    def save_model(self, saved_model_name:str, weights:bool=False):
        if weights:
            self.model.save_weights("../weights/" + saved_model_name + ".hdf5")
        else:
            self.model.save("../weights/" + saved_model_name + ".h5")


