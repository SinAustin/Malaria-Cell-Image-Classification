import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D

from keras.models import model_from_json
from keras.models import load_model

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

from sklearn.utils import shuffle
from skimage import transform
from skimage.color import rgb2gray
from skimage import io

import os
import glob



from IPython.display import clear_output



train_dir = './cell_images/train'
test_dir = './cell_images/test'


#import glob
#for img in glob.glob('cell_images/train/*.jpg'):
        #print (img.split('.')[0][-5])
        
 
# Our images now have the diagnosis in the file name. We will label our data based on the file name.
# files beginning with p are Parasitized cells
# files beginning with u are uninfected cells
# this will create our y/target variable 
def label_img(img):
    word_label=img.split('.')
    if word_label[1][-5]=='p':
        return [0,1]
    elif word_label[1][-5]=='u':
        return [1,0]        
        
# A function to create training data
def mk_train_data():
    train_data = []
    num = 0
    bad = 0
    for img in glob.glob('./cell_images/train/*.jpg'):
        # Using try because out of 26,000 images one could be bad
        try:
            # Labeling the cell image as parasitized or uninfected
            label = label_img(img)
            path = os.path.join(train_dir, img)
            
            # Loading the cell image, gray scaling it then resizing to our perfered input shape
            img = io.imread(img)
            img = rgb2gray(img)
            img = transform.resize(img,[60,60])
            
            #appending each image's data and label/classification in the form of arrays
            train_data.append([np.array(img), np.array(label)])
        except:
            bad+=1
            
        num+=1
        
        if num % 1000 == 0:
            print(num)
            print(bad)
    
    shuffle(train_data)
    # Saving the data so we don't have to create it again!
    np.save('train_data.npy', train_data)
    clear_output()
    print('Finished')
    return train_data        
    
    
def mk_test_data():
    test_data = []
    num = 0
    bad = 0
    for img in glob.glob('./cell_images/test/*.jpg'):
        try:
            
            label = label_img(img)
            path = os.path.join(test_dir, img)
            img = io.imread(img)
            img = rgb2gray(img)
            img = transform.resize(img,[60,60])
            test_data.append([np.array(img), np.array(label)])
        except:
            bad+=1
            
        num+=1
        
        if num % 100 == 0:
            print(num)
            print(bad)
    shuffle(test_data)
    np.save('test_data.npy', test_data)
    clear_output()
    print('Finished')
    return test_data    


# if creating train data for the 1st use
#train_data = mk_train_data()

# if creating test data for the 1st use 
#test_data = mk_test_data()

# Loading in the data we previously made    
train_data = np.load('./data/train_data.npy')
test_data = np.load('./data/test_data.npy')    
 
# Creating our train and test sets from our created datasets
X_train = np.array([i[0] for i in train_data]).reshape(-1, 60,60,1)
y_train = [i[1] for i in train_data]
y_train = np.squeeze(y_train)

X_test = np.array([i[0] for i in test_data]).reshape(-1, 60,60,1)
y_test = [i[1] for i in test_data]
y_test = np.squeeze(y_test)



# Listing any params here
imgsize = 60
eps = 5

# Building the Model*****

cnn_model = Sequential()

cnn_model.add(Conv2D(filters=32,
                     kernel_size=3,
                     activation='relu',
                     input_shape=[imgsize,imgsize,1] ))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

cnn_model.add(Conv2D(32,
                    kernel_size=3,
                    activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

cnn_model.add(Conv2D(64,
                    kernel_size=3,
                    activation ='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))


cnn_model.add(Flatten())
cnn_model.add(Dense(128,
                   activation='relu'))

cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(2,
             activation='sigmoid'))
             
# monitoring the training progress with tensor board
# creating a log folder
tensorboard = TensorBoard(log_dir='./logs',)

cnn_model.compile(loss='binary_crossentropy',
                 optimizer='Adam',
                 metrics=['accuracy'])
                 
# load Pretrained model if available
#loaded_model = load_model('cnn_model.hdf5')                 

#print('Model Loaded')

# training the model on our training data and checking the models performance on our testing data
history = cnn_model.fit(x=X_train,y=y_train,
                        validation_data=(X_test,y_test),
                        callbacks=[tensorboard],
                        epochs=eps,
                        verbose=1)
                        
# Saving our model and weights
cnn_model.save('cnn_model.hdf5',overwrite=True)
print('model saved')