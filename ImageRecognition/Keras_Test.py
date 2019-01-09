import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import backend as K
from keras.models import Sequential, load_model
K.set_image_dim_ordering('th')

#Load a presaved model. Load a test image from disk and prepare it for prediction. Visualize weights, and a couple of layers.
#Visualization based on https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb

load_model_name = "moj_model.h5"
load_model_path = os.getcwd() + "/save/"

# Load model 
model = load_model(os.path.join(load_model_path,load_model_name))
print('Model loading finished...displaying model summary')
model.summary()

#Load a test image from disk and prepare it for prediction
img_pred = PIL.Image.open("osam.jpg").convert("L")
img_array_pred = np.array(img_pred).reshape(1,1,28,28)
img_array_pred = img_array_pred.astype('float32')
img_array_pred /= 255

#Predict result
result = model.predict(img_array_pred,batch_size=1,verbose=1)

result_list = result[0]

print('Sorted results from the covnet')

new_order = [i for i in sorted(enumerate(result_list), key=lambda x:x[1], reverse=True)]

[print('{0}. {1:.16f}'.format(x[0],x[1])) for x in new_order]

#Display images from different layers

convout1 = model.get_layer('conv2d_1')
convout2 = model.get_layer('activation_1')
max_pooling1 = model.get_layer('max_pooling2d_1')

inputs = [K.learning_phase()] + model.inputs

_convout1_f = K.function(inputs, [convout1.output])
_convout2_f = K.function(inputs, [convout2.output])
_max_pooling1_f = K.function(inputs, [max_pooling1.output])


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
    pl.show(im)

def make_mosaic(imgs, nrows, ncols, border=1):
    
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border, ncols * imshape[1] + (ncols - 1) * border), dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border

    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols        
        mosaic[row * paddedh:row * paddedh + imshape[0], col * paddedw:col * paddedw + imshape[1]] = imgs[i]

    return mosaic

def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])

def convout2_f(X):
    # The [0] is to disable the training phase flag
    return _convout2_f([0] + [X])

def max_pooling1_f(X):
    # The [0] is to disable the training phase flag
    return _max_pooling1_f([0] + [X])


pl.figure()
pl.title('input')
nice_imshow(pl.gca(), np.squeeze(img_array_pred.reshape((28,28))), vmin=0, vmax=1, cmap=cm.binary)

# Visualize convolution result
C1 = convout1_f(img_array_pred)
C1 = np.squeeze(C1)
print("C1 shape : ", C1.shape)

pl.figure(figsize=(15, 15))
pl.suptitle('convout1')
nice_imshow(pl.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary)

C2 = convout2_f(img_array_pred)
C2 = np.squeeze(C2)
print("C2 shape : ", C2.shape)

pl.figure(figsize=(15,15))
pl.suptitle('convout2')
nice_imshow(pl.gca(),make_mosaic(C2,6,6),cmap=cm.binary)

MP = max_pooling1_f(img_array_pred)
MP = np.squeeze(MP)
print("MP shape : ", MP.shape)

pl.figure(figsize=(15,15))
pl.suptitle('max_pooling')
nice_imshow(pl.gca(),make_mosaic(MP,6,6),cmap=cm.binary)

# Visualize weights
W = model.layers[0].get_weights()
W = np.squeeze(W[0])
W = np.rollaxis(W,2,0)
print("W shape : ", W.shape)
pl.figure(figsize=(15, 15))
pl.title('conv1_weights')
nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)


WW = np.rollaxis(np.squeeze(model.layers[2].get_weights()[0][:,:,:,0]),2,0)
print("WW shape : ", WW.shape)
pl.figure(figsize=(15, 15))
pl.title('conv2_weights')
nice_imshow(pl.gca(), make_mosaic(WW, 6, 6), cmap=cm.binary)
