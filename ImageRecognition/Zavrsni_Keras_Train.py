import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import layers
from keras import models
K.set_image_dim_ordering('th')

# Training MNIST covnet based heavily on https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

batch_size = 128
nb_classes = 10
nb_epoch = 12

img_rows = 28
img_cols = 28

nb_filters = 32
nb_pool = 2
nb_conv = 3

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape, convert and normalize
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one hot vector
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# model
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid',input_shape = (1, img_rows, img_cols)))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes)) 
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])

# Learn
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

print('Training over.')

#Evalute model
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Saving the model')

save_model_name = "moj_model.h5"
save_model_path = os.getcwd() + "/save/"

model.save(os.path.join(save_model_path,save_model_name),include_optimizer = True)

print('Model is saved and ready for testing.')