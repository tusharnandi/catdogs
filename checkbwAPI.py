import sys
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from PIL import Image

class DelectBWNumber():
	
    def __init__(self):

        #Load Model and load Model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("model.h5")

        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
    def record(self, imageFile):
        
        self.imagefile = imageFile
        
        #print ("Image file: " + self.imagefile)
        #shape1: (28, 28)
        #shape2: (28, 28)
        #shape3: (28, 28, 1)
        #shape4: (1, 28, 28, 1)
        
        #print ("Starting: record function")
        img_array = cv2.imread(self.imagefile, 0)
        
        #print("shape-1:",img_array.shape)
        
        img_array = img_array.astype("float32") / 255
        #print("shape-2:",img_array.shape)   0-255
        
        img_array = np.expand_dims(img_array, -1)
        #print("shape-3:",img_array.shape)
        
        img_array = img_array.reshape(1, 28, 28, 1)
        
        #print("shape-4:",img_array.shape)
        #print(img_array)
        
        pred = self.loaded_model.predict(img_array)
        pnum = pred.argmax();
        return pnum;
############################################
#DelectBWNumber END


class DelectDogCat():
	
    def __init__(self):

        #Load Model and load Model
        base_path='./dogcatsModel/'
        json_file = open(base_path + "model_Resnet.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(base_path + "model_Resnet_lastweight.h5")
        self.loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def record(self, imageFile):
        dim = (224, 224)
        
        image = tf.keras.preprocessing.image.load_img(imageFile,target_size=dim)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        #print('Original Dimensions : ',input_arr.shape)
        
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        
        predictions = self.loaded_model.predict(input_arr)

        output=["cat","dog"]

        if (predictions > 0.5):
            prediction=1
        else:
            prediction=0
  
        return output[prediction]
        
############################################
#DelectBWNumber END
#================================
from flask import Flask
import os
from flask import render_template, flash, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER']= "./queueImage/"


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/upload')
def upload():
    """Renders the contact page."""
    return render_template(
        'upload.html',
        title='upload',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/dogcat')
def dogcat():
    """Renders the contact page."""
    return render_template(
        'dogcat.html',
        title='dogcat',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/dogcatdetect', methods = ['POST'])  
def dogcatdetect():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        imagePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        
        detectObj = DelectDogCat()
        predictionObj= detectObj.record(imagePath)

        return render_template("successdogcat.html", name = f.filename,prediction=predictionObj)  



@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  


        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

        imagePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        
        detectObj = DelectBWNumber()
        pnumber= detectObj.record(imagePath)

        return render_template("success.html", name = f.filename,predictedNumber=pnumber)  


@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )


#===============================
if __name__== "__main__":
    app.run(debug=True)