#from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import numpy as np
import os
from PIL import Image
from preprocessing import preprocess_image
from skimage.transform import resize
from tqdm import tqdm
import logging

logger = logging.getLogger ('data')
logger.setLevel (logging.INFO)

formatter = logging.Formatter ('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler ('../logs/data.log')
file_handler.setFormatter (formatter)

logger.addHandler (file_handler)



def get_train_data(pre_process=False):
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files('tawsifurrahman/covid19-radiography-database')
    zf = ZipFile('covid19-radiography-database.zip')
    #extracted data is saved in the same directory as notebook
    zf.extractall() 
    zf.close()

    folders = ['COVID','Lung_Opacity','Normal','Viral Pneumonia']
    base_folder = '/COVID-19_Radiography_Dataset'
    data = []
    labels = []

    for folder in folders :
      files = os.listdir(os.path.join(base_folder, folder))
      for file in files:
        img = Image.open(file)
        if(pre_process):
            img = preprocess_image(img)
        data.append(img)
        labels.append(folder)

    return np.array(data), np.array(labels)

def get_image(file):
    original_img = Image.open(file)
    loaded_image = np.array(original_img)
    logger.info ('The original image shape:{}'.format (loaded_image.shape))
    loaded_image = resize(loaded_image , (112 ,112))
    image = preprocess_image(loaded_image)
    logger.info ('Image is ready')
    return original_img, loaded_image, np.array(image)


"""
    load function for training
"""
def dataset(pre_process: bool = False, data_path: str = "../training/data/COVID-19_Radiography_Dataset"):
    
    folders = ['COVID','Lung_Opacity','Normal','Viral Pneumonia']
    # base_folder = '/COVID-19_Radiography_Dataset'
    data = []
    labels = []

    for folder in folders :
      files = os.listdir(os.path.join(data_path, folder))
      for file in tqdm(files):
        img = Image.open(os.path.join(data_path, folder, file))
        if(pre_process):
            img = preprocess_image(img)
        data.append(img)
        labels.append(folder)
    
    return np.array(data), np.array(labels)
