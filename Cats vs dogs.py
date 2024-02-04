#the code for training the machine learning model using CNN
import numpy as np
import pandas as pd
import cv2
import os #specify location
import matplotlib.pyplot as plt
import pickle
from PIL import Image 
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
Dir = "C:/Users/kalpa/Desktop/PetImages"
categ =['Cat','dog']
for c in categ:
    folder =os.path.join(Dir,c)
IMG_SIZE = 50  # Replace with your actual desired width
IMG_height = 50  # Replace with your actual desired height
data=[]
for c in categ:
    folder =os.path.join(Dir,c)
    label=categ.index(c)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
    # Check if img_arr is not None and has non-empty dimensions
        if img_arr is not None and img_arr.size != 0:
                img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_height))
                data.append([img_arr, label])
        else:
               print(f"Error reading or resizing image: {img_path}")
import random
random.shuffle(data)
x=[]
y=[]
for featu,labels in data:
    x.append(featu)
    y.append(labels)
x=np.array(x)
y=np.array(y)
model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128,input_shape=x.shape[1:],activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',#this is used when only 2 output neurons
             metrics=['accuracy'],run_eagerly=True)#want to research on optimisers
x_float = x.astype('float32')
model.fit(x_float,y,epochs=5,validation_split=0.3)
y_pred=model.predict(x)
