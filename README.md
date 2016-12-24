# Project 3 - Behavior Cloning

In this project we teach a car to drive on track using deep learning. The car is trained to drive on track 1 of Udacity simulator in
training mode and then tested on Autonomous mode in both track 1 and track 2 to check how well it learns for track1
and generalizes the learning for track2.


#### Architecture
* I have used a neural network consisting of 2 Convolution layer followed by 2 Fully Connected layers.
* I have used ELU activation after each layer except for the last layer. I have used ELU activation as it helps the model learn faster.
* I have also added MaxPooling after each Convolution layer to reduce number of features and only have the model learn the most important features.
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
* I have normalized images to between tange of -0.5 to 0.5
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

