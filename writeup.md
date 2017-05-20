#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

1. Submission files -- all project files are submitted to github, and available at [https://github.com/czechows/udacity----traffic-signs](https://github.com/czechows/udacity----traffic-signs)
The file Traffic_Sign_Classifier.html containst an html snapshot of the workbook.

2. Initial dataset summary:

 * Number of training examples = 34799
 * Number of validation examples = 4410
 * Number of testing examples = 12630
 * Image data shape = (32, 32, 3) (RGB)
 * Number of classes = 43

3. Exploratory visualization -- the ipython notebook contains exploratory visualization of the dataset, including a plot of the histogram of training examples,
and a display of one color image from each of the 43 classes of the images.

4. Preprocessing -- the preprocessing pipeline consists of

a) Normalization of pixels to range [-1,1]. This accelerates the training.

b) Conversion from RGB to grayscale. According to the paper [Sermanet, LeCun, 2011] this gives better performance on this set.

c) Creation of additional training images by the logic: if the image label is in set [9, 11, 12, 13, 15, 17, 18, 22, 26, 29, 30, 35],
then the image can be flipped horizontally to create a new training image with the same label. If the image label is in the set [12,15,17],
then the image can be flipped vertically to create a new training image with the same label. If the image label is in the set [33,36,38] 
then it can be flipped horizontally to create an image with a corresponding label from the set [34,37,39] and vice versa (so 33<->34 etc.).
The same image can be flipped both vertically and horizontally to create two additional training images. 
Increasing the number of training images makes the network generalize better and reduces the effect of overfitting.
As a result, the training set is augmented from 34799 examples to 53308 examples with relatively little effort.

5. Model architecture -- the neural network is a modified LeNet network, that consists of the following layers:

| Layer         		|     Description	        	| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 grayscale image      		        | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6	|
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16	|
| Flatten               | input 5x5x16 outputs 400      		|
| Dropout 		| keep prob. = 0.6			        |
| Fully connected	| input 400 outputs 250				|
| RELU			|						|
| Dropout 		| keep prob. = 0.6			        |
| Fully connected	| input 400 outputs 250				|
| RELU			|						|
| Dropout 		| keep prob. = 0.6			        |
| Fully connected	| input 250 outputs 120				|
| RELU			|						|
| Dropout 		| keep prob. = 0.6			        |
| Fully connected	| input 120 outputs 84				|
| RELU			|						| 
| Dropout 		| keep prob. = 0.6			        |
| Fully connected	| input 84 outputs 43 (logits)			|
| Argmax         	| input logits outputs predicted label		|
|:---------------------:|:---------------------------------------------:| 

Number of epochs: 120 -- Long, but still took shorter than creating artificial data. Additional epochs paid off in accuracy.
Batch size: 400 -- experimental choice.
Learn rate: 1e-3 -- left the same as for digit recognition.
Optimizer: Adam optimizer -- I did not compare with others, documentation online says that it's most cutting edge.
Mimimization of: cross entropy -- as is advised by several blogs to use it instead of MSE, performs better during backpropagation.

6. Solution approach

The initial accuracy on training set with LeNet network was around 0.89.
The LeNet network was chosen as a fundamental architecture for this problem,
as it performed well with digit recognition problems, thanks to its two convolutional layers.

By normalizing, adding an additional fully connected layer and converting to grayscale the network improved to around 0.93.
However, the validation accuracy was still below 0.9 (overfitting).
The problem was remedied by addition of dropout layers after the fully connected layers 
and by creation of additional data with use of symmetries.
The training accuracy improved to approx 0.97-0.98 and the validation accuracy to approx 0.97.
Initially, there was a bug in algorithm establishing validation accuracy and testing accuracy
(dropout was not disabled), which negatively affected the performance on validation set,
and appeared as if overfitting was still present.
Note to self: always disable dropout, and always make keep prob. a tf variable.

7. Acquiring new images

I acquired 5 new images of traffic signs: 
 * a 30km/h speed limit sign
 * a 60km/h speed limit sign
 * a children crossing sign
 * a stop sign
 * a turn left ahead sign.

The images are visualized in the iPython notebook and available in the web directory of github repo,
so there is no need to paste them here.

8. Performance on new images

Here are the results of the prediction:

| Image			        |     Prediction			| 
|:---------------------:|:---------------------------------------------:| 
| 30km/h        		| turn left ahead        		| 
| 60km/h     			| 60km/h 				|
| children crossing	  	| go straight or left			|
| stop	      	            	| stop					|
| turn left ahead		| turn left ahead			|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 
This is below the test accuracy, however, given that the images were not the part of the official German traffic sign set,
had watermarks and the traffic sign placement and size might have been different than most of the training set, I am relatively content with this result. 
30km/h was difficult to predict, as in the picture the sign was smaller, and shifted.
I have no good hypothesis on why I was unable to predict children crossing.

As a digression, at some point during feature testing I managed to achieve 80% accuracy by correctly predicting 30km/h,
however after refactoring the code and having to retrain the network, it was reduced to 60%.

9. Softmax probabilities

For the first image (30km/h) the probabilities are given below. One can see that the model was rather uncertain what to choose,
perhaps because the sign was smaller than usual and misplaced. The third guess was correct, however with a small probability.

| Probability         	|     Prediction	              		| 
|:---------------------:|:---------------------------------------------:| 
| .53         		| Stop sign   					| 
| .25     		| Keep right 					|
| .04			| 30km/h					|
| .03	      		| 80km/h				 	|
| .03		        | 20km/h      	        			|

For the second image (60km/h), the model had no issues, see table below

| Probability         	|     Prediction	              		| 
|:---------------------:|:---------------------------------------------:| 
| .97         		| 60km/h   					| 
| .02     		| 50km/h 					|
| .01			| 80km/h					|
| .00	      		| No passing				 	|
| .00		        | Ahead only   	        			|

For the third image (children crossing), the model mispredicted with high certainty. I am not sure why,
the signs don't look very similar.

| Probability         	|     Prediction	              		| 
|:---------------------:|:---------------------------------------------:| 
| .97         		| Straight or left				| 
| .00     		| Keep left 					|
| .00			| Ahead only					|
| .00	      		| Dangerous curve left  		 	|
| .00		        | Roundabout mandatory       			|

For the fourth image (stop), the model again did well. 

| Probability         	|     Prediction	              		| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         		| Stop   					| 
| .00     		| Straight or right 	        		|
| .00			| No entry					|
| .00	      		| Turn right ahead      		 	|
| .00		        | Priority road         			|

For the fifth, the prediction was also decisive

| Probability         	|     Prediction	              		| 
|:---------------------:|:---------------------------------------------:| 
| .99         		| Turn left ahead   				| 
| .01     		| Ahead only	                		|
| .00			| Keep right					|
| .00	      		| Go straight or right      		 	|
| .00		        | Priority road         			|



author: Aleksander Czechowski
