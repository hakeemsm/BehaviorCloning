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
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            correction = 0.1
            if len(batch_samples) == 0:
                    print("batch empty")
            for batch_sample in batch_samples:
                center_name = './drivinglog/IMG/'+batch_sample[0].split('/')[-1]
                left_name = './drivinglog/IMG/'+batch_sample[1].split('/')[-1]
                right_name = './drivinglog/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)         
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)       
                #print(center_image.shape)
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                measurements.append(center_angle)
                measurements.append(left_angle)
                measurements.append(right_angle)
                images.append(cv2.flip(center_image,1))
                measurements.append(center_angle*-1.0)
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)

validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
ch, row, col = 3, 80, 120
#model.add(Lambda(lambda x: x/127.7 - 1., input_shape = (ch, row, col), output_shape=(ch, row, col)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((50,20),(0,0))))

model.add(Conv2D(24, (5, 5), strides = (2,2), activation="relu"))

#model.add(ELU())

model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))

#model.add(ELU())

model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))

#model.add(ELU())

model.add(Conv2D(64, (3, 3), activation="relu"))

#model.add(ELU())

model.add(Conv2D(64, (3, 3), activation="relu"))

#model.add(ELU())

model.add(Flatten())

model.add(Dense(100))

model.add(Dropout(0.5))

#model.add(ELU())

model.add(Dense(50))

model.add(Dropout(0.5))

#model.add(ELU())

model.add(Dense(10))

model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,                     
                    steps_per_epoch = len(train_samples)/32,
                    epochs = 3,
                    validation_data = (validation_generator),              
                    validation_steps = len(validation_samples)/32)

model.save('model.h5')