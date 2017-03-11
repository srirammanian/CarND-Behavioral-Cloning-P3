#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/normal_driving.jpg "Normal Driving"
[image3]: ./examples/left_turn.jpg "Left turn Image"
[image4]: ./examples/left_turn_flipped.jpg "Flipped Image"
[image5]: ./examples/training_loss.png "Training Loss"
[image6]: ./examples/val_loss.png "Validation Loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
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

My model consists of a 5 layer convolutional network followed by a 4 layer fully connected network.  My model is based off the NVIDIA Self Driving car model with some slight tweaks to the filter size and strides in some of the layers to account for my resized input (I resized the input image from 320x160 to 160x80).  Please see model.py lines 146-164 for the architecture.

The input is resized by half using openCV upon load (line 64). The reason for resizing the input was to speed up training. I found the time per epoch dropped by around 50%. This input is passed into a Keras cropping layer (line 148) which removes the top 34% and bottom 15% of the image (accounts for sky and hood of car). Following this is a lambda layer which normalizes the input (line 149) to [-0.5,0.5].  After this is the 5 layer convolutional network, followed by a fully connected layer with RELUs.

[image1]

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 159-164). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 125-128). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer.  It was trained over 30 epochs, the first 10 at a learning rate of 0.001, and the last 20 at a rate of 0.0001.  

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and augmented my data with the sample data set provided.  I originally tried using my PS4 controller so that I could get better fine grained input but I had problems with the simulator reading the X-Y joystick values.  Therefore, most of the training data that I created was a combination of driving via the keyboard arrow keys and using my Apple mouse.  I found it difficult to use the Apple mouse and wasn't totally satisfied with the data which is why I included the sample set in my training data.  For the data I generated I drove 4 laps in the center of the lane, then did 1 lap where I tried to have as smooth curves as possible.  I considered driving data where I drove from the right and left parts of the lane and recovered to the middle, but I first tested to see how my model did without that data. It turned out the model did fine without needing that extra data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to first implement a simple convolutional network and then progress to using the NVIDIA Self Driving CNN architecture.  My first step was to create a simple model consistening of a single hidden layer so I could verify the  model and my code's correctness.  I tested this model with 3 images - a image where the steering input was going left, one going straight, and one going to the right.  Once I was able to get correct predictions I moved on to creating a simple convolutional network similar to the Lenet model used earlier in the course.  This model did better than I expected - it negotiated the first turn well but would get consistently stuck on the bridge.  At this point I decided to implement the NVIDIA CNN architecture recommended in the lesson. 
  In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I added dropout layers at 50% between each FC layer.  This solved the overfitting issue and flipped my loss with my training set loss being higher than my validation set. 
  At this point I used the left/right driving camera images as suggested by the lesson and found it made a huge difference in helping the model negotiate turns. I found I really did not need to augment the training data with my own recovery driving - though I'm sure this could make the model even better if it did end up on the edges of the road.
  My model now made it almost around the entire track, but had some issues with the left-right sequence of turns in the track.  It was at this point that I discovered my model was expecting a BGR format image (due to using openCV to load the image), but the drive.py code was using a different image reader and was inputting a RGB format image into my model.  I corrected drive.py to use BGR and then tested my model.  This change allowed the car to drive around the entire track without leaving the road!

####2. Final Model Architecture

The final model architecture can be found in model.py (lines 146-164). Here is a visualization of the architecture

![alt text][image1]

I adjusted the filter and stride size of the 2nd-5th layers of the CNN in order to accomodate my image resizing and also to optimize the number of parameters.  By using a stride 2 for the 2nd layer and using 3x3 filters the parameter size reduced from ~20k to ~6k parameters.  This reduction helped the model to train faster but did come at a price in some accuracy as the model is a little 'jerkier' in turns.  Still, the car was able to negotiate the entire track so I thought it was sufficient to keep the model as is.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would help the model better predict right hand turns as well (since track #1 is predominantly left hand turns). For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]


After the collection process, I had 14961 number of data points. I augmented this data with the sample driving set to have a total of 41346 images. After using flipped images, this becomes 68910 images.  I then preprocessed the data by resizing the images to by half in a lambda layer, and then cropping out the top and bottom portion of the image (sky and car).  

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by my training set not not monotonically improving its loss.  The following images show my training and validation loss:

![training loss][image5]
![validation loss][image6]
