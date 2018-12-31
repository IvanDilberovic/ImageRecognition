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
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import backend as K
import tensorflow as tf
K.set_image_dim_ordering('th')

#For 2D data (e.g.  image), "tf" assumes (rows, cols, channels) while "th"
#assumes (channels, rows, cols).
app = Flask(__name__)
CORS(app)

load_model_name = "moj_model_test.h5"
load_model_path = os.getcwd() + "/save/"
model = load_model(os.path.join(load_model_path,load_model_name))
model._make_predict_function() 
graph = tf.get_default_graph()
print('Model loaded, printing summary...')
model.summary()

global_image = None

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None, name=None,size=None):
        
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
        pl.savefig(byte_io,bbox_inches='tight', dpi=size) 
        byte_io.seek(0)
        base_64 = base64.b64encode(byte_io.read())
        base_64 = base_64.decode('utf-8')
                      
        return name,base_64

def make_mosaic(imgs, nrows, ncols, border=1):
        
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = np.zeros((nrows * imshape[0] + (nrows - 1) * border, ncols * imshape[1] + (ncols - 1) * border), dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border

    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols        
        mosaic[row * paddedh:row * paddedh + imshape[0], col * paddedw:col * paddedw + imshape[1]] = imgs[i]

    return mosaic

def set_cmap(layer):
    if layer == 'conv2d_1' or layer == 'conv2d_2':
        return cm.gray
    else:
        return cm.jet

def prepare_image(imageData):
    output = BytesIO(base64.b64decode(imageData))
    output.seek(0)

    image = Image.open(output).convert('L') #need to convert because the image comes as ARGB
    image.save(os.getcwd() + "/images/orginal.png")  
    image.thumbnail(size=(28,28),resample=Image.ANTIALIAS)
    image.save(os.getcwd() + "/images/resized.png")  
         
    img_for_prediction = np.array(image).reshape(1,1,28,28)
    img_for_prediction = img_for_prediction.astype('float32')
    img_for_prediction /= 255

    return img_for_prediction

@app.route('/')
def index():
    return "Hello world :)"

@app.route('/api/GetPrediction', methods=['GET', 'POST'])
def get_prediction():   
    
    try:
        imageData = request.json['slika'].split('base64,')[1]   
    
        img_for_prediction = prepare_image(imageData)
                
        global graph
        with graph.as_default():
            result = model.predict(img_for_prediction,batch_size=1,verbose=1)               
            listaRezultata = result[0]               
            list = []
            for i in sorted(enumerate(listaRezultata), key=lambda x:x[1], reverse=True):
                dic = {}
                dic["key"] = i[0]
                dic["value"] = '{0:.16f}'.format(i[1])
                list.append(dic)        
            
            return jsonify({'success': True, 'status_code': 200, 'message': '', 'results': list, 'images' : None})

    except Exception as e:
        print(str(e))
        return jsonify({'success': False, 'status_code': 500, 'message': str(e), 'results': None, 'images' : None})        

         
@app.route('/api/GetLayerImage', methods=['GET', 'POST'])    
def get_layer_image():

    try:

        layer = request.json['layer']
        image = request.json['slika'].split('base64,')[1]  
        image = prepare_image(image)
        message = ''
        list = []

        if model.get_layer(layer).output.shape.ndims == 2:
            message = 'Given layer does not have correct output dimensions so the image will not be created.'

        global graph
        with graph.as_default():
            inputs = [K.learning_phase()] + model.inputs
            _f = K.function(inputs,[model.get_layer(layer).output])        
            C1 = _f([0] + [image])
            C1 = np.squeeze(C1)
            x,y = nice_imshow(pl.gca(), make_mosaic(C1, 3, 11),cmap=set_cmap(layer), name=layer + ".png", size=400)
            d = {}
            d["name"] = x
            d["picture"] = y
            list.append(d)

        message = 'OK'       

        print("GetLayerImage()", message)
        
        return jsonify({'success': True, 'status_code': 200, 'message': message, 'results': None, 'images' : list})

    except Exception as ex:
         return print(str(ex))
         return jsonify({'success': False, 'status_code': 500, 'message': str(e), 'results': None, 'images' : None})

@app.route('/api/GetLayerNames', methods=['GET'])    
def get_layer_names():

    try:
        list = []
        message = ''

        for layer in model.layers:
            if layer.output.shape.ndims == 2:
                continue
            list.append(layer.name)
        
        message = 'OK'

        print("GetLayerNames()", message)

        return jsonify({'success': True, 'status_code': 200, 'message': message, 'results': list})

    except Exception as ex:
         return print(str(ex))
         return jsonify({'success': False, 'status_code': 500, 'message': str(e), 'results': None})


@app.route('/api/GetAllLayerImages', methods=['GET'])    
def get_all_layer_images():

    try:

        img_for_prediction = request.json['slika'].split('base64,')[1]
        img_for_prediction = prepare_image(img_for_prediction)

        list = []
        function_list = []
        
        global graph
        with graph.as_default():

            inputs = [K.learning_phase()] + model.inputs

            for layer in model.layers:

                print(layer.name)

                if model.get_layer(layer.name).output.shape.ndims == 2:
                    continue

                _f = K.function(inputs,[model.get_layer(layer.name).output])
                C1 = _f([0] + [img_for_prediction])
                C1 = np.squeeze(C1)
                x,y = nice_imshow(pl.gca(), make_mosaic(C1, 3, 11),cmap=set_cmap(layer), name=layer.name + ".png", size=400)
                d = {}
                d["name"] = x
                d["picture"] = y
                list.append(d)

        message = 'OK'

        print("GetAllLayerImages()", message)

        return jsonify({'success': True, 'status_code': 200, 'message': message, 'results': None, 'images' : list})

    except Exception as ex:
         return print(str(ex))
         return jsonify({'success': False, 'status_code': 500, 'message': str(e), 'results': None, 'images' : None})




@app.route('/api/GetWeightImage', methods=['GET'])    
def get_weight_image():

    try:    
       
        message = ''
        list = []
                
        global graph
        with graph.as_default():
             W1 = model.layers[0].get_weights()
             W1 = np.squeeze(W1[0])
             W1 = np.rollaxis(W1,2,0)
             x,y = nice_imshow(pl.gca(), make_mosaic(W1, 6, 6),cmap=cm.gray,name="Weights_1.png", size=50)            
             d = {}
             d["name"] = x
             d["picture"] = y
             list.append(d)
             
             #W2 = model.layers[2].get_weights()[0]
             #W2 = np.squeeze(W2)
             #W2 = W2.reshape((W2.shape[0], W2.shape[1], W2.shape[2] * W2.shape[3])) 
             ##W2 = np.rollaxis(W2,2,0)
             #x,y = nice_imshow(pl.gca(), make_mosaic(W2, 6, 6),cmap=cm.jet,name="Weights_2.png", size=400)            
             #d = {}
             #d["name"] = x
             #d["picture"] = y
             #list.append(d)

             ##W = np.squeeze(W)
             ##W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 
             ##fig, axs = plt.subplots(5,5, figsize=(8,8))
             ##fig.subplots_adjust(hspace = .5, wspace=.001)
             ##axs = axs.ravel()
             ##for i in range(25):
             ##    axs[i].imshow(W[:,:,i])
             ##    axs[i].set_title(str(i))

        message = 'OK'       

        print("GetWeightImage()", message)
        
        return jsonify({'success': True, 'status_code': 200, 'message': message, 'results': None, 'images' : list})

    except Exception as ex:
         return print(str(ex))
         return jsonify({'success': False, 'status_code': 500, 'message': str(e), 'results': None, 'images' : None})

if __name__ == '__main__':
    app.run(debug=False)
