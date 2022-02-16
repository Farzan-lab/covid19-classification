from models import EfficientNet, MobileNet


def load_model(model_name, pre_trained: bool = True):
    if model_name == 'efficientnet':
        model = EfficientNet(model_path='../weights/efficientnet.hdf5',
                               input_shape=(112, 112, 1),
                               num_classes=4,
                               pre_trained=pre_trained)
    
    elif model_name == 'mobilenet':
        model = MobileNet(model_path='../weights/mobilenet.hdf5',
                               input_shape=(112, 112, 1),
                               num_classes=4,
                               pre_trained=pre_trained)
    return model