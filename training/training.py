import sys
sys.path.append('../src')

from model_loader import *
from data import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.tensorflow
import argparse

parser = argparse.ArgumentParser(description="COVID classification")
# where is dataset?
parser.add_argument('-p', '--path', type=str, metavar='',
                    default="../training/data/COVID-19_Radiography_Dataset", 
                    help='Path to dataset [../training/data/COVID-19_Radiography_Dataset]')
# Original Images or pre-processed ones?
process = parser.add_mutually_exclusive_group()
process.add_argument('-pr', '--preprocess', action='store_true', help='Select Pre-Process')

# U-Net Or Residual U-Net?
network = parser.add_mutually_exclusive_group()
network.add_argument('-N', '--network', type=str, 
                    metavar='', default="mobilenet", 
                    help="Select Model['efficientnet', 'mobilenet']")
parser.add_argument('-su', '--summary', action='store_true', help="Print model summary") # print model summary

"""
    :hyper parameters: for training
"""
parser.add_argument('-bs', "--batch_size", type=int, metavar='', default=16, help='batch_size')
parser.add_argument('-e', "--epochs", type=int, metavar='', default=2, help='Number of epochs')
parser.add_argument('-H', '--history', action='store_true', help='Show history')

"""
    :mlflow: track all model resualts
"""
mlrecords = parser.add_mutually_exclusive_group()
mlrecords.add_argument('-ML', '--mlflow', action='store_true', help="Save Models with MLFlow")

"""
    :save model: save model for future
    :model name: a name for saved file
    :weights: save weights as hdf5 or save model as h5 
    :location: model will be saved in weights directory
"""
save_model = parser.add_mutually_exclusive_group()
save_model.add_argument('-S', '--save', action='store_true', help="Save Model")
parser.add_argument('-mn', '--modelName', type=str, metavar='', default="saved_model", help="Model Name for Saving")
parser.add_argument('-W', '--weights', action='store_true', help="Save Model's weights")

args = parser.parse_args()


def label_maker(Y_train, Y_test):
    encode = LabelEncoder()
    onehotencoder = OneHotEncoder()

    Y_train = encode.fit_transform(Y_train)
    Y_test = encode.transform(Y_test)
    train_labels = Y_train.reshape(-1, 1)
    test_labels = Y_test.reshape(-1, 1)
    
    Y_train = onehotencoder.fit_transform(train_labels)
    Y_train = Y_train.toarray()
    Y_test = onehotencoder.transform(test_labels)
    Y_test = Y_test.toarray()

    return Y_train, Y_test


if __name__ == "__main__":
    if args.mlflow:
        mlflow.autolog()
    
    # load dataset
    images, labels = dataset(pre_process=args.preprocess, data_path=args.path)
    images = images.reshape(-1, 112, 112, 1)
    images = images / 255.0

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, 
                                                    test_size = 0.15, 
                                                    random_state = 2018)

    Y_train, Y_test = label_maker(Y_train, Y_test)

    model = load_model(args.network, pre_trained=False)
    if args.summary:
        model.summary() ## print summary
    res = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch_size, epochs=args.epochs)

    if args.save:
        model.save_model(saved_model_name=args.modelName, weights=args.weights)

    # Show history for train and validation
    if args.history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
        ax1.plot(res.history['loss'], '-o', label = 'Loss')
        ax1.plot(res.history['val_loss'], '-o', label = 'Validation Loss')
        ax1.legend()

        ax2.plot(100 * np.array(res.history['acc']), '-o', 
                label = 'Accuracy')
        ax2.plot(100 * np.array(res.history['val_acc']), '-o',
                label = 'Validation Accuracy')
        ax2.legend()
        plt.show()
    