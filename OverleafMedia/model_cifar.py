# Dimensionalidad de la muestra
sample_shape = x_train[0].shape

# Construcción de la red
model = net_1(sample_shape, 10)
model.summary()

# Necesitamos compilar nuestro modelo de red neuronal
model.compile(loss='categorical_crossentropy',
              optimizer='Adagrad',
              metrics=['accuracy'])