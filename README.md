# Project 3 - Behavior Cloning

In this project we teach a car to drive on track using deep learning. The car is trained to drive on track 1 of Udacity simulator in
training mode and then tested on Autonomous mode in both track 1 and track 2 to check how well it learns for track1
and generalizes the learning for track2.


#### Approach taken to derive the Architecture for this Project
* First of all I analyzed the driving_log.csv file and studied the images captured in training mode.
* Just like the very first project of this course I realized that lot of data in the images is scenery
and can be masked out.
* Initially I masked the images by making top one-third portion black. I read some discussions on Udacity forums
and found that students had actually cropped out the images. I liked the idea but did not crop the images yet, instead
I started writing a rough model with this knowledge.
* For a model for this project I started with the basic Model I built at the end of Keras lab in Lesson 11 of this course.
https://github.com/anupriyachhabra/keras-lab/blob/master/traffic-sign-classification-with-keras.ipynb.
The final architecture of model is defined in section [Architecture](#architecture) .
* I decided to use AdamOptimizer as its is efficient and self tuning.
* Then I removed the softmax activation as this is a regression problem and softmax is good for classification problem.
* Also I changed the loss to "mean squared error" as that is the most common type of loss taught in this course
 and also recommended by Francois Chollet for regression problems here https://github.com/fchollet/keras/issues/108
* When calling the model.fit method with all the images my program started crashing, so I thought to actually crop the images
to one-third size. But on further analysis of images I decided to be bold and crop top half of the image.
* Cropping the top half of the image made my model compile and I was able to test out my first implementation in autonomous
mode. My actually drove quite well except for sharp turns, where it got stuck.
* I started with 32 filters of size 3X3 for each conv net. I did increase the number of filters from 32 to 64 to 256 and saw
improved performance with that. Increasing the number of filters increased the training time a lot but the performance improvement
was not that great and I had the belief that this is a simple set of images , its only a video game not a real world scenario
and I should be able to train a model with less number of filters. So I continued with 32 filters.
* Till this point I was only training my model for 3 epochs, I decided to train it for more epochs- upto 20 and saved the weights
using callbacks - ModelCheckPoint.
* I then applied the saved weights in drive.py and ran the model against random epochs starting with weights for 3rd epoch,
5th epoch etc. I saw no improvement after 10th epoch so I settled on 10 for number of epochs.
* At this point my car was still failing to make sharp turns but driving smoother.
* I explored using transfer learning using VGG or AlexNet, but I thought that the dataset used in these models is very
different and I should try to improve my model.
* I studied the images again and decided to add the left and right images with an adjustment to steering angle to compensate
for the side cameras. A visualization of camera angles is given in section [Training](#training).
*  After adding the left and right images my model started failing at model.fit method due to three times the number of images.
At that point I went ahead and wrote a custom batch_generator so that I can use model.fit_generator to compile the model in
batches. At this point the the model started compiling.
* The compiled model started making turns neatly but at the last right turn it was detecting
right turn as straight in few areas. So I removed further scenery from right and left of image to make the model more sure
of what it is supposed to be doing. A detailed explanation of this approach is given in section [PreProcessing and Image Cropping](#preprocessing-and-image-cropping)
* At this point my model started making all correct predictions but the angles were not big enough for turns. So it was
turning left and right where it was supposed to but the predicted steering angle was falling a bit short, and my car was
touching lane lines.
* Till this point I had not added any activations to my model and decided to add "tanh" activation to my model as this activation
produces stronger gradients. I equated that to stronger angles if the input is large and smoother angles if input is small.
I referred this article https://theclevermachine.wordpress.com/tag/tanh-function/
* After adding the tanh activation my model started making stronger turns as desired, but the mse got higher than before.
That is when I looked at ELU activation and found that it has many properties similar to tanh but is better than tanh since it helps
the model learn faster. So I decided to use ELU in final submission.



#### Architecture
* I have used a neural network consisting of 2 Convolution layer followed by 2 Fully Connected layers.
* Both the conv nets consists of 32 filters of size 3X3 and stride 1 with valid padding.
* I have used ELU activation after each layer except for the last layer. I have used ELU activation as it helps the model learn faster.
* I have also added 4X4 MaxPooling after each Convolution layer to reduce number of features and only have the model learn the most important features.
* I have added dropout 0.5 after 1st Conv layer and 0.75 after 2nd Conv layer to reduce overfitting.

* Following is a detailed visualization of the model

![Model Architecture](model.png?raw=true "Model Architecture")


#### Training

* For this project I have used data provided by Udacity.
* Model was trained on a MacBook Pro (Retina, 15-inch, Mid 2015) with Udacity Simulator
version 5.4.2f2 on screen resolution 640 x 480 and fastest graphics quality
* I have also used left and right camera images as I was not getting a smoothly running car model with just center images
* An adjustment of +0.25 has been made to the steering angle for left camera image and added to the training labels,
so that if the car sees a scene similar to left camera it can find the appropriate steering angle.
* Similar to above an adjustment of -0.25 has been made to right camera image.

|Center Image | Left Image | Right Image|
|-------------|------------| -----------|
|![Center Image](example_images/center_2016_12_01_13_33_46_039.jpg?raw=true)|![Left Image](example_images/left_2016_12_01_13_33_46_039.jpg?raw=true)|![Right Image](example_images/right_2016_12_01_13_33_46_039.jpg?raw=true)|
|Steering Angle = 0.34|Steering Angle = 0.69|Steering Angle = 0.09|


* Also I have used a custom generator to feed data to model.fit_generator as the images were not fitting the RAM of my computer.



#### PreProcessing and Image Cropping
* I have normalized images to between range of -0.5 to 0.5
* When I was using only center images, I cropped top half of the image as it
does not contain data relavant to track, its just scenery. Size of my image was reduce to 80 x 320
* After I added left and right images I realised that the car bonnet is an extra parameter that the model has to learn,
so I removed last 25 rows from images and added those 25 rows to top keeping the image size same as above 80 x 320
* After all the above preprocessing my model was still sometimes detecting right turn as straight so I cropped 50 columns
from left and right of the image so that if the model is supposed to turn right it sees very less or no left markers
 - Eg - cropped image for turning right
   ![Cropped Image](example_images/turn_right_cropped.jpg?raw=true)

 - Original uncropped image
   ![Uncropped Image](example_images/turn_right_uncropped.jpg?raw=true)

 - In the uncropped image above there are lot of lane markers on right but significant amount in left as well, making the
 model make wrong predictions. In the cropped image the model has more amount of right line markers and significantly less
 left markers so model can make prediction to turn right with more accuracy.


#### Simulation and Driving
* I have reduced the throttle of car to 0.1 when making turns so that it does not oversteer.
* Also I have increased throttle to 0.4 if speed goes less than 5.0, this was only needed for track 2



#### Extra Library installed
* I have used Keras Visualization library in visualization.py to produce a visualization of this model
as shown in the architecture section.
* visualization.py produces a file called model.png which has model structure.
* I have installed pydot and its related dependencies to use Keras Visualization as shown below using
pip install
- pydot
- pydotplus
- graphviz
- pydot-ng
if graphviz gives errors - brew install graphviz
https://github.com/fchollet/keras/issues/3210

