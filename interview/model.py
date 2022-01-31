# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 19:05:01 2021

@author: USER
"""
import numpy as np
import tensorflow as tf
import os 
import matplotlib.cbook as cbook 
import cv2
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook 
n=100
data_test=[]
test_big_list=os.listdir("test/big")
test_small_list=os.listdir("test/small")
home=home=os.getcwd()
way=[home+'/train/big/',home+'/train/small/',home+'/test/big/',home+'/test/small/']
for i in test_big_list:
    with cbook.get_sample_data(way[2]+i) as image_file:
        image = plt.imread(image_file)
        image= cv2.resize(image,(n,n))
        image = (image/255.0).astype(np.float32)
        data_test.append((image,1))

for i in test_small_list:
    with cbook.get_sample_data(way[3]+i) as image_file:
        image = plt.imread(image_file)
        image= cv2.resize(image,(n,n))
        image = (image/255.0).astype(np.float32)
        data_test.append((image,0))


def model_body():
    input_tensor=tf.keras.Input(shape=(n,n,3,))
    x=tf.keras.layers.RandomFlip()(input_tensor)
    x=tf.keras.layers.RandomRotation(0.25)(x)
    x=tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model= tf.keras.Model(input_tensor,x,name="Classifier")
    return model
model=model_body()

model.load_weights(home +"/save_model/")


x_test=np.zeros((len(data_test),n,n,3))
y_test=np.zeros(len(data_test))

for elem in range(len(data_test)) :
    q=data_test.pop(0)
    x_test[elem]=q[0]
    y_test[elem]=q[1]
predict=model.predict(x_test)
predict=tf.transpose(tf.round(predict))
print(predict)
print(y_test)
print(y_test-predict)