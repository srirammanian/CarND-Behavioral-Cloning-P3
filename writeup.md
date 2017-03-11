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
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and augmented my data with the sample data set provided.  I originally tried using my PS4 controller so that I could get better fine grained input but I had problems with the simulator reading the X-Y joystick values.  Therefore, most of the training data that I created was a combination of driving via the keyboard arrow keys and using my Apple mouse.  I found it difficult to use the Apple mouse and wasn't totally satisfied with the data which is why I included the sample set in my training data.  For the data I generated I drove 3 laps in the center of the lane, then I did 2 laps of data recovering from left and ride sides of the road, then did 1 lap where I tried to have as smooth curves as possible.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to first implement a simple convolutional network and then progress to using the NVIDIA Self Driving CNN architecture.  My first step was to create a simple model consistening of a single hidden layer so I could verify the  model and my code's correctness.  I tested this model with 3 images - a image where the steering input was going left, one going straight, and one going to the right.  Once I was able to get correct predictions I moved on to creating a simple convolutional network similar to the Lenet model used earlier in the course.  I thought this model might be appropriate due to 

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
