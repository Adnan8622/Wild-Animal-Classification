# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:02:19 2021

@author: DeepLearning
"""

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from imutils import paths
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.optimizers import Adam,SGD
from keras.preprocessing.image import img_to_array
import efficientnet.keras as efn
from matplotlib import pyplot

path_test='D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/test_data/'
#path_test='E:/Desktop data/Wild animal classification using ML/Dataset/'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(path_test,target_size=(224, 224),
												batch_size=32,class_mode='categorical',shuffle = False)

from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
num_of_test_samples = len(test_generator)
target_names = ["gazella","giraffe ","hartebeest","hippopotamus","Lion","wildbeest","Buffalo","elephant","warthog ","zebra"]
#num_of_test_samples=1785
#Confution Matrix and Classification Report
model = load_model('D:/MS Proposal/Research Code/Proposed Results/Best Results final itteration/EfficientNetB0.h5')

y_pred = model.predict(test_generator)
y_pred = np.round(y_pred).astype(int)

y_pred = np.argmax(y_pred, axis=1)

#pred_cls_name = CLASSES_LIST[y_pred]
#print("percentage values",Y_pred)
#print(y_pred)
#print('warthog',y_pred[0])
#print('giraffe',y_pred[1])
#print('hartebeest',y_pred[2])
#show some images from the dataset and put predicted labels on it

x,y = test_generator.next()
a=len(x)
print(a)
c=0
for i in range(a):
    # define subplot
    #pyplot.subplot(330 + 1 + i)
    print(c)
    c=c+1
    image = x[i]
    label = y[i]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #print ('Actual Labels',label)
    #print('predicted Labels ',y_pred[i])
    lab=y_pred[i]
    #print(image.shape)
    #print(target_names[i],lab)
    pred_cls=target_names[lab]
    print(pred_cls)
    if pred_cls=='gazella':
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Gazella/gazella'+str(i)+'.jpg', (image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if pred_cls=='hartebeest':
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Hartebeest/hartebeest'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if pred_cls=='wildbeest':
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Wildbeest/wildbeest'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if pred_cls=='elephant':
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Elephant/elephant'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if pred_cls=='hippopotamus':
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Hippopotamus/hippopotamus'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if pred_cls=='Lion':
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Lion/Lion'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if pred_cls=='zebra':
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Zebra/zebra'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if lab==1:#Geraffi
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Giraffe/giraffe'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if pred_cls=='Buffalo':
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Buffalo/Buffalo'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
    if lab==8:#Warthog
        cv2.imwrite('D:/MS Proposal/MS Thesis/Thesis Chapters/Research Code/Predicted classes/Warthog/warthog'+str(i)+'.jpg',(image*255))
        #cv2.imshow('',image)
        #cv2.waitKey(0)
     
    cv2.putText(image,target_names[int(lab)], (10,image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # plot raw pixel data
   # pyplot.imshow(image)
# show the figure
#pyplot.show()

print(len(y_pred))
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
print(cm)
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

import matplotlib.pyplot as plt
import matplotlib.pyplot 
from mlxtend.plotting import plot_confusion_matrix
plt.figure(figsize=(20, 20))
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False,
                                class_names=target_names)

plt.savefig("EfficientNetB0_Confusion Matrix")
plt.show()

import seaborn as sns
import pandas as pd
cmn = cm.astype('float') / cm.sum(axis=1)#neaxwis used to conver row into column
fig, ax = plt.subplots(figsize=(20,7))

sns.heatmap(cmn, center=0, annot=True, fmt='.2f', linewidths=1,  xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
plt.savefig("EfficientNetB0_Confusion_Matrix")
plt.show()
"""
plt.figure(figsize=(40,40))
sn.set(font_scale=2.4) # for label size
sn.heatmap(cm, annot=True, annot_kws={"size":20}) # font size
plt.savefig("EfficientNetB0 Matrix 10 classes")
plt.show()
"""
