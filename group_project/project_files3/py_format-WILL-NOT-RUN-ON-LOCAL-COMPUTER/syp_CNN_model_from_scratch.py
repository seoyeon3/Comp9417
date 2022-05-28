#!/usr/bin/env python
# coding: utf-8
##################################################################
# This code shows a scratch CNN model tested as initial trial.
##################################################################
#################################
# import modules
#################################
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torchvision.transforms as tr
import pandas as pd


#################################
# load train samples from npy file
#################################
X_train=np.load("X_train.npy", mmap_mode='r') 
y_train=np.load("y_train.npy", mmap_mode='r')


#################################
# reduce sample size
#################################
resize = []
for i in range(858):
    x_resize = cv2.resize(X_train[i],(256,256))
    resize.append(x_resize)


x_resize = np.array(resize)


#################################
# make a dataloader
#################################
class TensorData(Dataset):
    
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.x_data = self.x_data.permute(0,3,1,2) # img, channel, width, height
        self.y_data = torch.LongTensor(y_data)
        self.len = self.y_data.shape[0]
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.len


train_data = TensorData(resize, y_train)


train_loader = DataLoader(train_data, batch_size = 300, shuffle = True)


len(train_loader) #we have 858 images and batch size is 300. So we got 3


dataiter = iter(train_loader)
images, labels = dataiter.next()


images.size() #we have 100 images


#################################
# set the random seed for reproduction
#################################
np.random.seed(1)


#################################
# randomly extract 30 imgs from 100 imgs
#################################
index_list = np.arange(0,len(images))
valid_index = np.random.choice(index_list, size = 100, replace = False)


valid_images = images[valid_index] #30x3x1024x1024
valid_labels = labels[valid_index]


train_index = set(index_list)-set(valid_index)
train_images = images[list(train_index)]
train_lables = labels[list(train_index)]


#################################
# print the images to check
#################################

plt.figure(figsize=(20,20))
for i in range(70):
    plt.subplot(15,15,i+1)
    plt.imshow(train_images[i]) 
    plt.title(f'class={int(train_lables[i])}') 
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

#################################
# import modules
#################################
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, TensorBoard

#################################
# prepare the train x and y samples
#################################
x_train = train_images.numpy()
y_train = train_lables.numpy()

x_valid = valid_images.numpy()
y_valid = valid_labels.numpy()


#################################
# model preparation
#################################
model = tf.keras.Sequential()
model.add(Conv2D(6, kernel_size=(3,3), activation = 'relu', padding='same', input_shape = (256,256,3)))
model.add(Conv2D(16, kernel_size=(3,3), activation = 'relu', padding='same', input_shape = (256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(4, activation='softmax'))


model.summary()


#################################
# compile the model
#################################
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy']  
             )


call_backs = [
    EarlyStopping(patience=3, monitor='val_loss'),
    TensorBoard(log_dir='./logs',histogram_freq=1)
]


#################################
# train the model
#################################
model_fit = model.fit(x_train, y_train,
                      batch_size = 40, 
                      epochs = 30,
                      validation_data = (x_valid, y_valid),
                      callbacks=call_backs
)


#################################
# see the result on Tensorboard
#################################
get_ipython().run_line_magic('load_ext', 'tensorboard')


get_ipython().run_line_magic('tensorboard', '--logdir logs')


#################################
# get the prediction from trained model
#################################
prediction = model.predict(x_valid)


predictions = []
for i in range(len(prediction)):
    predictions.append(np.argmax(prediction[i]))

predictions = np.array(predictions)


#################################
# check the final result in form of dataframe
#################################
df = pd.DataFrame({'y_predic':predictions, 'y_true':y_valid})
df

