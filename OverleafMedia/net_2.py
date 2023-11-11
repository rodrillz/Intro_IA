def net_2(sample_shape, nb_classes):
     #Defina la entrada de la red para que tenga la dimensión `sample_shape`
    input_x = Input(shape=sample_shape)

    # Generar 32 kernel maps utilizando una capa convolucional
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_x)

    # Generar 64 kernel maps utilizando una capa convolucional con stride=2
    x = Conv2D(64, (3, 3), padding='same', activation='relu', strides=2)(x)

    # Aplanar el feature map
    x = Flatten()(x)

    # Capa fully-connected, 128 dimensiones con activación ReLU
    x = Dense(128, activation='relu')(x)

    # Capa fully-connected a nb_classes dimensiones con activación Softmax
    probabilities = Dense(nb_classes, activation='softmax')(x)

    # Definir la salida
    model = Model(inputs=input_x, outputs=probabilities)

    return model