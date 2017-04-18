# **Traffic Sign Recognition** 

### Data Set Summary & Exploration

[//]: # (Image References)

[image1]: ./images/plot1.png 
[image2]: ./images/plot2.png 
[image3]: ./images/preprocessing.png 
[image4]: ./images/test_imgs.png 
[image5]: ./images/top_5_Softmax.png 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2.Visualization of the dataset.

I have used the matplot library to plot the two histograms 
one for train, test and vaildation images and the other for each Unique labels.

![alt text][image1]

![alt text][image2]

Also I have ploted random training images for better understanding of data.


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.


I have applied a very simple preprocessing, as a first step, I calcuated the mean and substracted every images from the mean to make them zero centered. And to make all data in same ranage I have normalized the images by dividing them with standard deviation of zero centered image.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image3]



#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

To cross validate my model, data was already divided into traing, test and validataion data.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 24x24x16	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 20x20x16	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 16x16x16	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x10	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x10 					|
| Fully connected		| input 360 output 120							|
| RELU					|												|
| Fully connected		| input 120 output 84							|
| RELU					|												|
| Fully connected		| input 84 output 43							|
| RELU					|												|
|						|												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
        
To train the model, I used an Adam optimizer , batch size of 128 and 5 epochs and learning rate of 0.001.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I have followed the iterative process, at first I tried experimenting around the LeNet architecture from the lab example with additin of dropout layer, this architeture gave me validation accuracy of 93.4%, this in the function CNet1. Later I have tried adding few more layers (CNet2), I have added a additional Conv layer which increased a validation accuracy for 95% in only 5 epocs.

My final model results were:
* training set accuracy of 99.2% 
* validation set accuracy of 96.1%
* test set accuracy of 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    
	First arcitecture was based on LeNet architecture
    
* What were some problems with the initial architecture?
		
	The accuracy was very low arround 83 %
    
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    
	I Expermented the dropout and maxpool layer which resulted in drastic increase in accuracy to 95%. From this we can say that adding dropout and maxpool prevented the overfitting of data.
	And added more convolution layers
    
* Which parameters were tuned? How were they adjusted and why?

Experimented around different values for learning rate
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

	Maxpool layer helps deal with over fitting of data.

If a well known architecture was chosen:
* What architecture was chosen?
	I have adopted and experimented on LeNet and AlexNet.

* Why did you believe it would be relevant to the traffic sign application?
	I did some homwork on these Neural nets, given that these were devloped for image net challenge we can say that these would be relevant to traffic sign application 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	With iterative devlopement the accuracy of model is about 95% on validataion data, this provides evidence that the model is working well
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Keep right									| 
| Bumpy road   			| General caution								|
| No entry  			| No entry 										|
| Speed limit (50km/h)	| Speed limit (50km/h)			 				|
| Dangeus curve to right| End of all speed and passing limits			|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

 ![alt text][image5]

