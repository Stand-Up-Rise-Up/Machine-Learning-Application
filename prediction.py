#!/usr/bin/python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image
import cv2
import os
from flask import Flask, render_template, request

print("Import Success")

app = Flask(__name__)

json_model = open("xray_resnet50v2_1.json",'r')
loaded_model_json = json_model.read()
json_model.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("xray_resnet50v2_weights.h5")
print('loaded model')
model = loaded_model

label = {0 : "COVID", 1 : "Lung Opacity", 2 : "Normal", 3 : "Viral Pneumonia"}

@app.route('/',methods=['GET'])

def hello_world():
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files["imagefile"]
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = cv2.imread(image_path)
    width, height = 299,299
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image = np.reshape(resized,[1,299,299,3])
    image = image/255.

    yhat = model.predict(image)
    max_prob = np.max(yhat)
    output = np.argmax(yhat, axis = 1)

    classification = '%s  (%.2f%%)' % (label[output[0]], max_prob*100)

    
    return render_template("index.html", prediction = classification)




if __name__ == '__main__':
    app.run(port=3000, debug= True)