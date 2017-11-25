import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, ELU, Dropout, Cropping2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

samples = []

#Read image locations and angles from csv
with open('./drivinglog/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Train and test split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    """
    Generator to lazy yield a set of samples at a time.
    samples: set of training data
    batch_size: batch size to process and return
    The method shuffles the samples and batches them. For each batch
    1. Get the left, center and right images to append to the result set
    2. Get the steering angles for left, center and right, apply correction and add to returned set
    """
    num_samples = len(samples)
    img_path = './drivinglog/IMG/'  #path containing the images
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            correction = 0.1
            
            for batch_sample in batch_samples:
                center_name = img_path + batch_sample[0].split('/')[-1]
                left_name = img_path + batch_sample[1].split('/')[-1]
                right_name = img_path + batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)         #read center image
                left_image = cv2.imread(left_name)             #read left image 
                right_image = cv2.imread(right_name)           #read right image 
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction         #apply correction to left angle 
                right_angle = center_angle - correction        #apply correction to right measurement 
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                measurements.append(center_angle)
                measurements.append(left_angle)
                measurements.append(right_angle)
                images.append(cv2.flip(center_image,1))        #flip the center image and append 
                measurements.append(center_angle*-1.0)         #measurement for the flipped image 
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)      #shuffle and yield sample 

train_generator = generator(train_samples, batch_size=32)      #call generator for training data 

validation_generator = generator(validation_samples, batch_size=32) #call generator for validation data

model = Sequential()    #create model

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))) #lambda layer for normalizing images

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))   #crop images to remove extraneous features

model.add(Conv2D(24, (5, 5), strides = (2,2), activation="relu"))   #layer1 ConvNet

model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))     #layer2 ConvNet

model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))     #layer3 ConvNet

model.add(Conv2D(64, (3, 3), activation="relu"))                    #layer4 ConvNet

model.add(Conv2D(64, (3, 3), activation="relu"))                    #layer5 ConvNet

model.add(Flatten())    #flatten the output from previous layer

model.add(Dense(100))   #fully connected layer1

model.add(Dropout(0.5)) #dropout1 

model.add(Dense(50))    #fully connected layer2

model.add(Dropout(0.5)) #dropout2

model.add(Dense(10))    #fully connected layer3

model.add(Dense(1))     #fully connected layer4

model.summary()

model.compile(loss='mse', optimizer='adam') #train the model with Adam optimizer

#fit model and visualize loss
history_object = model.fit_generator(train_generator,                     
                    steps_per_epoch = len(train_samples)/32,
                    epochs = 3,
                    validation_data = (validation_generator),              
                    validation_steps = len(validation_samples)/32, verbose=1)

print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#save model to disk
model.save('model.h5')