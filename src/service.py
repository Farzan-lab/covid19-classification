import streamlit as st
import numpy as np
import numpy as np
import keras
from data import get_image
from models import EfficientNet, MobileNet
from gradcam import GradCAM
import logging
from resource_manager import *

logger=logging.getLogger("service")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s %(message)s')
file_handler = logging.FileHandler('../logs/service.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def show_class(label):
    if label==0:
        st.write(f'the prediction is: COVID-19')
    if label==1:
        st.write(f'the prediction is: Lung Opacity')
    if label==2:
        st.write(f'the prediction is: Normal')
    if label==3:
        st.write(f'the prediction is: Viral Pneumonia')

st.write("COVID-19 prediction")

#html input with an extra extension checker
file = st.file_uploader("Please upload the chest x-ray image file", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Please upload an image file:")
else:

    logger.info ('An image file loaded')

    original_img, loaded_img, image = get_image(file)
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    # Prepare image
    img_array = preprocess_input(image)

    st.text("preprocessing done!")
    st.image(original_img, use_column_width=True)

    st.text("MobileNet Prediction:")
    mobile_net = MobileNet(model_path='../weights/mobilenet.hdf5',
                               input_shape=(112, 112, 1),
                               num_classes=4)
    mobile_net.model.layers[-1].activation = None

    pred = mobile_net.predict(image)
    
    # monitor CPU and memory usage
    resources = Resource_Manager()
    resources = resources.monitor()

    label = np.argmax(pred)
    show_class(label)
    model =  mobile_net.model.get_layer('mobilenetv2_1.00_224')
    last_conv_layer_name = "out_relu"
    gradcam = GradCAM(model, last_conv_layer_name, loaded_img)
    img_array = preprocess_input(gradcam.get_img_array())
    heatmap = gradcam.make_gradcam_heatmap(img_array)

    # monitor CPU and memory usage
    resources = Resource_Manager()
    resources = resources.monitor()

    st.image(gradcam.save_and_display_gradcam(original_img,heatmap), use_column_width=True)

    st.write('class probs:')
    st.write(pred)

    st.text("EfficientNet Prediction:")
    efficient_net = EfficientNet(model_path='../weights/efficientnet.hdf5',
                               input_shape=(112, 112, 1),
                               num_classes=4)
    pred = efficient_net.predict(image)

    # monitor CPU and memory usage
    resources = Resource_Manager()
    resources = resources.monitor()
    
    label = np.argmax(pred)
    show_class(label)

    model =  efficient_net.model.get_layer('efficientnetb0')
    last_conv_layer_name = "top_activation"
    
    gradcam = GradCAM(model, last_conv_layer_name, loaded_img)
    
    # monitor CPU and memory usage
    resources = Resource_Manager()
    resources = resources.monitor()
    
    img_array = preprocess_input(gradcam.get_img_array())
    heatmap = gradcam.make_gradcam_heatmap(img_array)

    st.image(gradcam.save_and_display_gradcam(original_img,heatmap), use_column_width=True)
   
    st.write('class probs:')
    st.write(pred)
