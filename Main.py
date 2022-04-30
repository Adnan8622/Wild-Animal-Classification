# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import img_to_array
import keras 

from keras import layers
from keras import models

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
"""
path='dataset'

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 2
INIT_LR = 1e-3
BS = 12
IMAGE_DIMS = (224, 224, 3)

# grab the image paths and randomly shuffle them
print("loading images...")
imagePaths = sorted(list(paths.list_images(path)))
random.seed(42)
random.shuffle(imagePaths)
# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
	
print(labels)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
print(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.30, random_state=42)
print("Train X = ",trainX.shape)
print("Test X = ",testX.shape)
print("Train Y = ",trainY.shape)
print("Test Y = ",testY.shape)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

from models import alexnet
model = alexnet((224,224,3),2)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print(" training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
model.save('facial_model.h5')
import matplotlib.pyplot as plt
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('facial_plot.png')
plt.show()
"""

# -*- coding: utf-8 -*-


import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Model

path_train='train data/'

#train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,rotation_range=25, width_shift_range=0.1,
#	height_shift_range=0.1,fill_mode="nearest")
train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.30,rotation_range=15)
test_datagen = ImageDataGenerator(rescale=1./255)

#train generator and validation generator
train_generator = train_datagen.flow_from_directory(path_train,target_size=(224, 224),subset = 'training',
												batch_size=32,class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(path_train,target_size=(224, 224),subset='validation',
												batch_size=32,class_mode='categorical',shuffle = False)
#test_generator = test_datagen.flow_from_directory(path_train,target_size=(224, 224),
#												batch_size=32,class_mode='categorical',shuffle = False)

image_size = 224
IMG_SHAPE = (image_size, image_size, 3)
batch_size = 32
 
from models import alexnet,our_model,VGG_16,inception_v3,VGG_19,EfficientNetB0
from VGG_Models import VGG16,VGG19 
#Create the base model from the pre-trained model MobileNet V2
#model = alexnet(IMG_SHAPE,2)
model = EfficientNetB0(IMG_SHAPE,10) # image size and Number of classes

#compiling model             
model.compile(optimizer=keras.optimizers.SGD(), #Adam(lr=0.00001)
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('Trained_Model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
callbacks=[es, mc]

#Trainng the model
epochs = 50
steps_per_epoch = 32
validation_steps = 32

history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              callbacks = callbacks)

#saving model
model.save('Trained_Model.h5')  # creates a HDF5 file 'my_model.h5'
print('model saved!')

#plotting the graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, 'b-+', label='Training Accuracy')
plt.plot(val_acc,'r-*', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.savefig("Training and Validation Accuracy Graph")

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.savefig("Training and Validation Loss")
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
num_of_test_samples = len(validation_generator)
target_names = ["gazella","giraffe ","hartebeest","hippopotamus","Lion","wildbeest","Buffalo","elephant","warthog ","zebra"]
#num_of_test_samples=1785
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print(len(y_pred))
print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred)
print(cm)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

import matplotlib.pyplot as plt
import matplotlib.pyplot 
from mlxtend.plotting import plot_confusion_matrix
plt.figure(figsize=(20, 20))
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False,
                                class_names=target_names)

plt.savefig("Confusion Matrix_2nd_iteration")
plt.show()

import seaborn as sns
import pandas as pd
cmn = cm.astype('float') / cm.sum(axis=1)#neaxwis used to conver row into column
fig, ax = plt.subplots(figsize=(20,7))

sns.heatmap(cmn, center=0, annot=True, fmt='.2f', linewidths=1,  xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
plt.savefig("2nd Confusion Matrix_2nd_iteration")
plt.show()

