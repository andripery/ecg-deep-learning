from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Upload path
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIIONS = {'csv'}

# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dataset path
DATASET_PATH = 'datasets/data_test__kachuee__default_ovr_train__29-06-2020_05-38-23.csv'

# Model saved with Keras model.save()
MODEL_PATH = 'models/kachuee__default_ovr_train__29-06-2020_05-38-23.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIIONS

def model_predict(file_path, model):
    datatest = np.loadtxt('uploads/'+file_path, delimiter=',', skiprows=1)
    x_test = datatest[:,0:260]
    y_test = datatest[:,260]

    preds = np.argmax(model.predict(x_test), axis=1)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    classes = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No File')
        # Get the file from post request
        file = request.files['file']

        # Save the file to ./uploads
        if file.filename == '':
            flash('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Make prediction
        preds = model_predict(filename, model)

        # Process your result for human
        result = preds[0]

        if result == 0:
            classes = 'N'
        elif result == 1:
            classes = 'V'
        elif result == 2:
            classes = 'S'
        elif result == 3:
            classes = 'F'
        else: 
            classes = 'Q'

    return classes

if __name__ == '__main__':
    app.run(debug=True)

