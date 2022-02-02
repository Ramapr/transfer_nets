from keras.engine.input_layer import Input
from keras.layers import Flatten, Dense, Dropout, Reshape
from keras.layers import Conv2D, MaxPooling2D, AvgPool2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50, DenseNet169, DenseNet201





def resnet50(weights_path):
    den = ResNet50(input_shape=(224, 224, 3), include_top=False, weights=None, input_tensor=None)
    den.load_weights(weights_path, by_name=True)
    x = GlobalAveragePooling2D()(den.output)
    out = Dense(1, activation='sigmoid')(x)
    return  Model(den.input, out)
