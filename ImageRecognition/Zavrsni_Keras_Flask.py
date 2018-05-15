from flask import Flask, jsonify, abort, request, Response, make_response, send_file
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image, ImageMath
from io import BytesIO
import io
import pylab as pl
from scipy.misc import imsave, imread, imresize
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import json
from keras.models import Sequential, load_model
from keras import utils
from sklearn import preprocessing
import matplotlib.cm as cm
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import backend as K
K.set_image_dim_ordering('th')

#For 2D data (e.g.  image), "tf" assumes (rows, cols, channels) while "th"
#assumes (channels, rows, cols).
app = Flask(__name__)
CORS(app)

load_model_name = "moj_model.h5"
load_model_path = os.getcwd() + "/save/"
model = load_model(os.path.join(load_model_path,load_model_name))
print('Load modela gotov, printam summary...')
model.summary()

#Function for saving images from layers
def SaveLayerOutput(img_for_prediction):

    def conv2d_1_function(X):    
        return _conv2d_1_f([0] + [X])

    def activation_1_function(X):    
        return _activation_1_f([0] + [X])

    def conv2d_2_function(X):    
        return _conv2d_2_f([0] + [X])

    def activation_2_function(X):    
        return _activation_2_f([0] + [X])

    def max_pooling2d_1_function(X):    
        return _max_pooling2d_1_f([0] + [X])

    def dropout_1_function(X):    
        return _dropout_1_f([0] + [X])

    #def deprocess_image(x):
    ## normalize tensor: center on 0., ensure std is 0.1

    #    with np.errstate(divide="raise"):
            
    #        x -= x.mean()
    #        x /= (x.std() + 1e-5)
    #        x *= 0.1
            
    #        # clip to [0, 1]
    #        x += 0.5
    #        x = np.clip(x, 0, 1)
            
    #        # convert to RGB array
    #        x *= 255
    #        x = x.transpose((1, 0))
    #        x = np.clip(x, 0, 255).astype('uint8')
    #        x = np.nan_to_num(x)
    #        return x

    def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None, name=None,size=None):
        """Wrapper around pl.imshow"""
        if cmap is None:
            cmap = cm.jet
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
                                    
        im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap) 

        path = os.path.dirname(os.getcwd()) + "/" + "ImageRecognitionWebApp" + "/" + "Images"

        ax.yaxis.set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_visible(False)
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        byte_io = io.BytesIO()
        #pl.savefig(buf,os.path.join(path,name), bbox_inches='tight', dpi=size) #maknuo dpi=size
        pl.savefig(byte_io,bbox_inches='tight', dpi=size) 
        byte_io.seek(0)
        base_64 = base64.b64encode(byte_io.read())
        base_64 = base_64.decode('utf-8')

        #im = Image.open(buf)
        #im.show()
        
        #im.figure.gca().yaxis.set_visible(False)
        #im.figure.gca().xaxis.set_visible(False)

        #im.figure.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())        
        #im.figure.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                                      
        #im.figure.canvas.draw()
 
        ## Get the RGBA buffer from the figure
        #w,h = im.figure.canvas.get_width_height()
        #buf = np.fromstring(im.figure.canvas.tostring_argb(), dtype=np.uint8)
        #buf.shape = (w,h,4)
 
        ## canvas.tostring_argb give pixmap in ARGB mode.  Roll the ALPHA
        ## channel to have it in RGBA mode
        #buf = np.roll(buf, 3, axis = 2) 

        #w, h, d = buf.shape

        #pikcr = Image.frombytes("RGBA", (w ,h), buf.tostring())

        #pikcr.show()

        
        return name,base_64
        #return os.path.basename(os.path.normpath(name))

    def make_mosaic(imgs, nrows, ncols, border=1):
        """
        Given a set of images with all the same shape, makes a
        mosaic with nrows and ncols
        """
        nimgs = imgs.shape[0]
        imshape = imgs.shape[1:]
    
        #mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
        #ncols * imshape[1] + (ncols - 1) * border), dtype=np.float32)
        mosaic = np.zeros((nrows * imshape[0] + (nrows - 1) * border, ncols * imshape[1] + (ncols - 1) * border), dtype=np.float32)
    
        paddedh = imshape[0] + border
        paddedw = imshape[1] + border

        for i in range(nimgs):
            row = int(np.floor(i / ncols))
            col = i % ncols        
            mosaic[row * paddedh:row * paddedh + imshape[0], col * paddedw:col * paddedw + imshape[1]] = imgs[i]

        return mosaic

    

    conv2d_1 = model.get_layer('conv2d_1')
    activation_1 = model.get_layer('activation_1')
    conv2d_2 = model.get_layer('conv2d_2')
    activation_2 = model.get_layer('activation_2')
    max_pooling2d_1 = model.get_layer('max_pooling2d_1')
    dropout_1 = model.get_layer('dropout_1')  


    inputs = [K.learning_phase()] + model.inputs

    _conv2d_1_f = K.function(inputs, [conv2d_1.output])
    _activation_1_f = K.function(inputs, [activation_1.output])
    _conv2d_2_f = K.function(inputs, [conv2d_2.output])
    _activation_2_f = K.function(inputs, [activation_2.output])
    _max_pooling2d_1_f = K.function(inputs, [max_pooling2d_1.output])
    _dropout_1_f = K.function(inputs, [dropout_1.output])

    lista = []
    
    C1 = conv2d_1_function(img_for_prediction)
    C1 = np.squeeze(C1)
    x,y = nice_imshow(pl.gca(), make_mosaic(C1, 3, 11),cmap=cm.binary, name="Conv2d_1.png", size=400)
    d = {}
    d["name"] = x
    d["picture"] = y
    lista.append(d)

    W = model.layers[0].get_weights()
    W = np.squeeze(W[0])
    W = np.rollaxis(W,2,0)
    nice_imshow(pl.gca(), make_mosaic(W, 6, 6),cmap=cm.gray, name="Weights_1.png", size=50)

    A1 = activation_1_function(img_for_prediction)
    A1 = np.squeeze(A1)
    x,y = nice_imshow(pl.gca(), make_mosaic(A1, 3, 11), cmap=cm.jet, name="Activation_1.png", size=400)
    d = {}
    d["name"] = x
    d["picture"] = y
    lista.append(d)

    C2 = conv2d_2_function(img_for_prediction)
    C2 = np.squeeze(C2)    
    x,y = nice_imshow(pl.gca(), make_mosaic(C2, 3, 11), cmap=cm.gray, name="Conv2d_2.png", size=400)
    d = {}
    d["name"] = x
    d["picture"] = y
    lista.append(d)

    A2 = activation_2_function(img_for_prediction)
    A2 = np.squeeze(A2)    
    x,y = nice_imshow(pl.gca(), make_mosaic(A2, 3, 11), cmap=cm.jet, name="Activation_2.png", size=400)
    d = {}
    d["name"] = x
    d["picture"] = y
    lista.append(d)

    MP = max_pooling2d_1_function(img_for_prediction)
    MP = np.squeeze(MP)    
    x,y = nice_imshow(pl.gca(), make_mosaic(MP, 3, 11), cmap=cm.jet, name="MaxPooling_1.png", size=400)
    d = {}
    d["name"] = x
    d["picture"] = y
    lista.append(d)

    DP = dropout_1_function(img_for_prediction)
    DP = np.squeeze(DP)    
    x,y = nice_imshow(pl.gca(), make_mosaic(DP, 3, 11), cmap=cm.jet, name="Dropout_1.png", size=400)
    d = {}
    d["name"] = x
    d["picture"] = y
    lista.append(d)

    return lista    

@app.route('/')
def index():
    return "Hello world :)"

@app.route('/api/GetPrediction', methods=['GET', 'POST'])
def get_prediction():   

    try:
        imageData = request.json['slika'].split('base64,')[1]   
    
        output = BytesIO(base64.b64decode(imageData))
        output.seek(0)
        image = Image.open(output).convert('L') #need to convert because the image comes as ARGB
        image.save(os.getcwd() + "/images/orginal.png")  
        image.thumbnail(size=(28,28),resample=Image.ANTIALIAS)
        image.save(os.getcwd() + "/images/resized.png")  
         
        img_for_prediction = np.array(image).reshape(1,1,28,28)
        img_for_prediction = img_for_prediction.astype('float32')
        img_for_prediction /= 255
    
        result = model.predict(img_for_prediction,batch_size=1,verbose=1)

        listaRezultata = result[0]
               
        list = []

        for i in sorted(enumerate(listaRezultata), key=lambda x:x[1], reverse=True):
            dic = {}
            dic["key"] = i[0]
            dic["value"] = '{0:.16f}'.format(i[1])
            list.append(dic)    
    
        lista = SaveLayerOutput(img_for_prediction)    

        return jsonify({'success': True, 'status_code': 200, 'message': '', 'results': list, 'images' : lista})

    except Exception as e:
        print(str(e))
        return jsonify({'success': False, 'status_code': 500, 'message': str(e), 'results': None, 'images' : None})        

@app.route('/api/GetImages')
def get_images():

     try:
         byte_io = BytesIO()
         image = Image.open(os.path.dirname(os.getcwd()) + "/ImageRecognitionWebApp/Images/" + "Activation_1.png") 
         image.save(byte_io, 'PNG')
         byte_io.seek(0)
         base_64 = base64.b64encode(byte_io.read())
         base_64 = base_64.decode('utf-8')

         #return
         #send_file(base_64,'image/png')#mimetype='application/octet-stream'
         return jsonify({'success': True, 'status_code': 200, 'message': '', 'results': None, 'images' : base_64})

     except Exception as ex:
         return print(str(ex))
         return jsonify({'success': False, 'status_code': 500, 'message': str(e), 'results': None, 'images' : None})
         
    
   
if __name__ == '__main__':
    app.run(debug=False)
