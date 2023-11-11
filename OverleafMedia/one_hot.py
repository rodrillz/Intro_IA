y_train = one_hot(y_train.reshape(-1), 10)
y_test = one_hot(y_test.reshape(-1), 10)


### *No* modifique las siguientes líneas ###
# Imprima los tamaños de los datos (variables)
print('Shape of x_train {}'.format(x_train.shape))
print('Shape of y_train {}'.format(y_train.shape))
print('Shape of x_test {}'.format(x_train.shape))
print('Shape of y_test {}'.format(y_train.shape))