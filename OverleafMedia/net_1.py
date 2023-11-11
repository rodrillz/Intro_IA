# Importar la librería Keras
import keras
from keras.models import Model
from keras.layers import *


def net_1(sample_shape, nb_classes):
    # Defina la entrada de la red para que tenga la dimensión `sample_shape`
    input_x = Input(shape=sample_shape)

    # Generar 32 kernel maps utilizando una capa convolucional
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_x)

    # Generar 64 kernel maps utilizando una capa convolucional
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Reducir los feature maps utilizando max-pooling
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Aplanar el feature map
    x = Flatten()(x)

    # Capa fully-connected, 128 dimensiones con activación ReLU
    x = Dense(128, activation='relu')(x)

    # Capa fully-connected a nb_classes dimensiones con activación Softmax
    probabilities = Dense(nb_classes, activation='softmax')(x)

    # Definir la salida
    model = Model(inputs=input_x, outputs=probabilities)

    return model