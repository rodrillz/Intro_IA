def normalize_images(images):
  images = images.astype('float32') / 255.0  # Normaliza las imágenes dividiendo entre 255

  return images

### *No* modificar las siguientes líneas ###
test_normalize_images(normalize_images)

# Normalizar los datos para su uso futuro
x_train = normalize_images(x_train)
x_test = normalize_images(x_test)

def one_hot(vector, number_classes):
    """Devuelve una matriz codificada one-hot dado el vector argumento.
    """
    # Aquí almacenaremos nuestros one-hots
    one_hot= []

    # Aquí se codifica el 'vector' one-hot
    for val in vector:
        one_hot1 = [0] * number_classes  # Inicializar un array de ceros con longitud number_classes
        one_hot1[val] = 1  # Establecer el valor en 1 en la posición indicada por el valor en el vector
        one_hot.append(one_hot1)  # Agregar el one-hot actual a la lista

    # Transformar la lista en una matriz numpy y retornarla
    return np.array(one_hot)


### *No* modifique las siguientes lineas ###
test_one_hot(one_hot)

# One-hot codifica los labels de MNIST
y_train = one_hot (y_train, 10)
y_test = one_hot(y_test, 10)