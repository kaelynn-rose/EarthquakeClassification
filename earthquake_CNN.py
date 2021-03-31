import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from skimage import io, color, filters
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
import cv2

import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import cartopy
from cartopy import config
import cartopy.crs as ccrs

from datetime import datetime

################### USER INPUT ####################

# path to image dataset
dir = 'images/toy_data/specs_alt'

# paths to csv and hdf5 (waveform/signal) files
noise_csv_path = 'data/chunk1/chunk1.csv'
noise_sig_path = 'data/chunk1/chunk1.hdf5'
eq1_csv_path = 'data/chunk2/chunk2.csv'
eq1_sig_path = 'data/chunk2/chunk2.hdf5'
eq2_csv_path = 'data/chunk3/chunk3.csv'
eq2_sig_path = 'data/chunk3/chunk3.hdf5'
eq3_csv_path = 'data/chunk4/chunk4.csv'
eq3_sig_path = 'data/chunk4/chunk4.hdf5'
eq4_csv_path = 'data/chunk5/chunk5.csv'
eq4_sig_path = 'data/chunk5/chunk5.hdf5'
eq5_csv_path = 'data/chunk6/chunk6.csv'
eq5_sig_path = 'data/chunk6/chunk6.hdf5'

# read the noise and earthquake csv files into separate dataframes:
earthquakes_1 = pd.read_csv(eq1_csv_path)
earthquakes_2 = pd.read_csv(eq2_csv_path)
earthquakes_3 = pd.read_csv(eq3_csv_path)
earthquakes_4 = pd.read_csv(eq4_csv_path)
earthquakes_5 = pd.read_csv(eq5_csv_path)
noise = pd.read_csv(noise_csv_path)

full_csv = pd.concat([earthquakes_1,earthquakes_2,earthquakes_3,earthquakes_4,earthquakes_5,noise])

################## END USER INPUT ###################


# read the noise and earthquake csv files into separate dataframes:
earthquakes_1 = pd.read_csv(eq1_csv_path)
earthquakes_2 = pd.read_csv(eq2_csv_path)
earthquakes_3 = pd.read_csv(eq3_csv_path)
earthquakes_4 = pd.read_csv(eq4_csv_path)
earthquakes_5 = pd.read_csv(eq5_csv_path)
noise = pd.read_csv(noise_csv_path)

full_csv = pd.concat([earthquakes_1,earthquakes_2,earthquakes_3,earthquakes_4,earthquakes_5,noise])

# filtering the dataframe: uncomment if needed
#df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
#print(f'total events selected: {len(df)}')

# create list of traces in the image datset
traces_array = []
for filename in os.listdir(dir):
    if filename.endswith('.png'):
        traces_array.append(filename[12:-4])

# select only the rows in the metadata dataframe which correspond to images
img_dataset = full_csv.loc[full_csv['trace_name'].isin(traces_array)]
labels = img_dataset['trace_category'] # target variable, 'earthquake' or 'noise'
labels = labels.map(lambda x: 1 if x == 'earthquake_local' else 0) # transform target variable to numerical categories
labels = np.array(labels)
len(img_dataset)

# create an array of all images
imgs = []
for i in range(0,len(img_dataset['trace_name'])):
    img= cv2.imread(dir+'/fig_specalt_'+img_dataset['trace_name'].iloc[i]+'.png',0) # read in image as grayscale image
    imgs.append(img)
imgs = np.array(imgs)
imgs.shape

# train test split on images, 75% training data and 25% testing data
train_images, test_images, train_labels, test_labels = train_test_split(imgs,labels,random_state=41)
print(train_images.shape)
print(train_labels.shape)
train_images = train_images/255.0 # scale intensity to between 0 and 1
test_images = test_images/255.0 # scale intensity to between 0 and 1

img_height = 100
img_width = 100

train_images = train_images.reshape(-1,img_height,img_width,1)
test_images = test_images.reshape(-1,img_height,img_width,1)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='./saved_models/waves_toydataset_epoch{epoch}',
        save_freq='epoch')
]

# build CNN on toy 60000 sample dataset
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation = 'relu', padding = 'same'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten(input_shape=(110,110)))
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
model.fit(train_images,train_labels,epochs=5,callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("\nTest data, accuracy: {:5.2f}%".format(100*test_acc))

from datetime import datetime
saved_model_path = "./saved_models/waves_toydataset_5epochs_0.9935testacc_{}.h5".format(datetime.now().strftime("%Y%m%d")) # _%H%M%S
# Save entire model to a HDF5 file
model.save(saved_model_path)

predicted_classes = np.argmax(model.predict(test_images),axis=-1) # predict the class of each image
predicted_probs = model.predict(test_images) # predict the probability of each image belonging to a class


# create confusion matrix
cm = confusion_matrix(test_labels,predicted_classes) # compare target values to predicted values and show confusion matrix
print(cm)
accuracy = accuracy_score(test_labels,predicted_classes)
precision = precision_score(test_labels,predicted_classes)
recall = recall_score(test_labels,predicted_classes)
print(f'The accuracy of the model is {accuracy}, the precision is {precision}, and the recall is {recall}.')

# plot confusion matrix
plt.style.use('default')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not earthquake','earthquake'])
disp.plot(cmap='Blues', values_format='')
plt.tight_layout()
plt.show()









class Seismic():

    def __init__(self,full_csv,dir):
        self.full_csv = full_csv
        self.dir = dir
        self.traces_array = []
        self.img_dataset = []
        self.labels = []
        self.imgs = []
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = []
        self.test_loss = []
        self.test_acc = []
        self.predicted_classes = []
        self.predicted_probs = []

        # create list of traces in the image datset
        print('Creating seismic trace list')
        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                self.traces_array.append(filename[12:-4])

        # select only the rows in the metadata dataframe which correspond to images
        print('Selecting traces matching images in directory')
        self.img_dataset = self.full_csv.loc[self.full_csv['trace_name'].isin(self.traces_array)]
        self.labels = self.img_dataset['trace_category'] # target variable, 'earthquake' or 'noise'
        self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0)) # transform target variable to numerical categories
        print(f'The number of traces in the directory is {len(self.img_dataset)}')

        # create an array of all images
        print('Creating array of images for list of traces in directory')
        count = 0
        for i in range(0,len(self.img_dataset['trace_name'])):
            count += 1
            print(f'Working on trace # {count}')
            img= cv2.imread(dir+'/fig_specalt_'+self.img_dataset['trace_name'].iloc[i]+'.png',0) # read in image as grayscale image
            self.imgs.append(img)

        self.imgs = np.array(self.imgs)
        print('Done creating images array')
        print(f'The shape of the images array is {self.imgs.shape}')


    def train_test_split(self,test_size,random_state):
        # train test split on images, 75% training data and 25% testing data
        train_images, test_images, train_labels, test_labels = train_test_split(self.imgs,self.labels,random_state=random_state,test_size=test_size)
        print(f'The training images set is size: {train_images.shape}')
        print(f'The training labels set is size: {train_labels.shape}')
        print(f'The testing images set is size: {test_images.shape}')
        print(f'The testing labels set is size: {test_labels.shape}')
        
        print('Scaling image intensity')
        train_images = train_images/255.0 # scale intensity to between 0 and 1
        test_images = test_images/255.0 # scale intensity to between 0 and 1

        img_height = train_images.shape[1]
        img_width = train_images.shape[2]

        print('Resizing images')
        self.train_images = train_images.reshape(-1,img_height,img_width,1)
        self.test_images = test_images.reshape(-1,img_height,img_width,1)
        self.train_labels = train_labels
        self.test_labels = test_labels

    def train_cnn(self,epochs):
    
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='./saved_models/waves_toydataset_epoch{epoch}',
                save_freq='epoch')
        ]

        # build CNN on toy 60000 sample dataset
        print('Building CNN model')
        
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation = 'relu', padding = 'same'))
        model.add(keras.layers.MaxPool2D(2,2))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten(input_shape=(110,110)))
        model.add(keras.layers.Dense(16,activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10,activation='softmax'))

        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
        model.fit(self.train_images,self.train_labels,epochs=epochs,callbacks=callbacks)

        print(model.summary())
        
        # Set model save path
        saved_model_path = "./saved_models/waves_toydataset_5epochs_{}.h5".format(datetime.now().strftime("%Y%m%d")) # _%H%M%S
        # Save entire model to a HDF5 file
        model.save(saved_model_path)
        
        self.model = model


    def evaluate_model(self):
        print('Evaluating model on test dataset')
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=1)
        print("\nTest data, accuracy: {:5.2f}%".format(100*test_acc))

        print('Finding predicted classes and probabilities to build confusion matrix')
        self.predicted_classes = np.argmax(self.model.predict(self.test_images),axis=-1) # predict the class of each image
        self.predicted_probs = self.model.predict(self.test_images) # predict the probability of each image belonging to a class

        # create confusion matrix
        print('Building confusion matrix')
        cm = confusion_matrix(self.test_labels,self.predicted_classes) # compare target values to predicted values and show confusion matrix
        print(cm)
        accuracy = accuracy_score(self.test_labels,self.predicted_classes)
        precision = precision_score(self.test_labels,self.predicted_classes)
        recall = recall_score(self.test_labels,self.predicted_classes)
        print(f'The accuracy of the model is {accuracy}, the precision is {precision}, and the recall is {recall}.')

        # plot confusion matrix
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not earthquake','earthquake'])
        disp.plot(cmap='Blues', values_format='')
        plt.tight_layout()
        plt.show()
        


# Using the class
s = Seismic(full_csv,dir)
s.train_test_split(img_height=100,img_width=100,test_size=0.25,random_state=41)
s.train_cnn(epochs=5)
s.evaluate_model()
