#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project were as follows:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* A written report with a summary of the performance


[//]: # (Image References)

[image1]: ./examples/center.png "Center camera image"
[image2]: ./examples/left.png "Left Image"
[image3]: ./examples/right.png "Right Image"
[image4]: ./examples/normal.png "Normal Image"
[image5]: ./examples/flipped.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 which is a recording of the driving in autonomous mode
* drivinglog folder that contains the images and a csv file that includes the image names and steering angles
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of 5 ConvNets with kernel sizes ranging from 5x5 to 3x3 and four fully connected layers with units starting from 100 down to 1 (model.py lines 62-92) 

The model includes RELU layers to introduce nonlinearity after each ConvNet (code lines 68-76), and the data is normalized in the model using a Keras lambda layer (code line 64). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 82 & 86). 

The model was trained and validated on different data sets captured from training with varying number of loops to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by adding a small correction to the left and right angles

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was

A CNN similar to LeNet for MNIST but with more ConvNets and fully connected layers. This approach yielded good results for the German traffic signs classification with minimal overfitting and validation loss

The image and steering angle data was split into training and validation sets (line 20). The model was overfitting when only the center images and angles  were used for training. It improved vastly after I took the left and right angles into consideration and added a small correction to each

Tweaking the epoch and batch sizes also helped with overfitting issues. I started out with 10 epochs intially and observed an overfit. The model was refined by training various numbers for epoch and finally settled with 3 epochs as it gave good performance overall


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I captured more training data and also swerved the car left and right bringing back to the center so that the model can observe different angles

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 62-92) consisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Layer1:
|   Input         		| 160x320x3 RGB image   						| 
|   Convolution         | 2x2 stride, valid padding 	                |
|   Activation		    | RELU											|
| Layer2:
|   Convolution 	    | 2x2 stride, valid padding                     |
|   Activation          | RELU                                          |
|   Flatten             |                                               |
| Layer3:
|   Convolution 		| 2x2 stride, valid padding        				|
|   Activation          | RELU                                          |
| Layer4:
|   Convolution 		| 1x1 stride, valid padding      				|
|   Activation          | RELU                                          |
| Layer5:
|   Convolution 		| 1x1 stride, valid padding         			|
|Flatten				|           									|
| Layer6:				|												|
|	Fully connected		| 100 units										|
|Dropout
|  Layer7:				|												|
|	Fully connected		| 50 units										|
|Dropout
|  Layer8:				|												|
|	Fully connected		| 10 units										|
|  Layer9:				|												|
|	Fully connected		| 1 unit										|

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to move back to center when encoungtering a curve in the road. These images capture the left and right camera angles for a recovery

![alt text][image2]
![alt text][image3]


To augment the data sat, I also flipped images and angles so that the training data is captured from all angles. For example, here is an image that has then been flipped:

![alt text][image5]

and here is the original image

![alt text][image4]

After the collection process, I had 3936 data points. I then preprocessed this data by cropping the images and normalizing them (lines 64 & 66)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the validation loss and the vehicle always being in the center of the road while driving I used an adam optimizer so that manually training the learning rate wasn't necessary.
