# Advanced Machine Learning Project: Feature Visualization

### [Group Member]
+ Yili Tang 
+ Ziqi Liu 
+ Chenfei Jiang 
+ Jingtian Yao 

## Part 1: MNIST model and visualization

### plot_images
plot_images(images, cls_true)
		*Plot 9 images in a 3x3 grid, and write the true and predicted classes below each image.*
Parameters:
		images: array_like
				An array that can represent grayscale images, with shape (n, width, height), n here is the number of images.
		cls_true: array_like
				The true clusters of n images.

### plot_conv_output
plot_conv_output(vals)
		*Plot output of the convolutional layers.*
Parameters:
		values: array_like
				An array of the output of a convolutional layer with 4 dimensions, with shape (batch_size, width, height, #filters).

## Part 2: Visualization on VGG16 model of Imagenet Dataset

### normalize
normalize(x)
		*Normalize a tensor by its L2 norm.*
Parameters:
		x: A gradients tensor.		
		
### deprocess_image
deprocess_image(x)
		*Normalize an image: center on 0, ensure std is 0.1.*
Parameters:
		x: array_like
				An array that can represent a color image.

### saliency_map
saliency_map(layer_name,iter = 20, n_filters = 200,  n = 8, 
                			 img_width = 224, img_height = 224,
               		     layder_dict = layer_dict,save = True)
		*Draw the saliency map for a convolutional layer in VGG16 model.*
Parameters:
		layer_name: str
				The name of a convolutional layer given when training the model.
		iter: int
				Number of times of iteration to run gradient ascent to maximize the loss function.
		n_filters: int
				Number of filters to be scanned through.
		n: int
				The width and height of the grid that shows the result.
		img_width: int
				The width of the images that are used to train the model.
		img_height: int
				The height of the images that are used to train the model.
		layer_dict: dict
				A dictionary. The keys are the layer names of the model, the values are objects like keras.layers.convolutional.Conv2D etc.
		save: bool
				If true, the saliency map would be saved in the machine.

### initial
initial(index)
		*Initialize the image of the saliency map of softmax dense layer with the mean value of all the training images of a class on each pixel.*
Parameters:
		index: int
				The index of a class (from 0 to 999) on Imagenet dataset.
				
### visualize_softmax_dense_VGG			
visualize_softmax_dense_VGG(model, iter, output_index = 0, step = 0.5,
                                							alpha = 0.1, initialize = 'random',
                              						    initialize_index = 0, save = True)
		*Draw the saliency map for the softmax dense layer in VGG16 model.*
Parameters:
		model: 
				A keras training model.
		iter: int
				Number of times of iteration to run gradient ascent to maximize the loss function.
		output_index: int
				The index of a class (from 0 to 999) on Imagenet dataset.
		step: float
				The step of gradient ascent.
		alpha: float
				The regulization parameter of the loss function.
		initialize: str {'random','mean'}
				The initialization method of the saliency map. If random, initialize with a random noise; if mean, initialize with the mean value of all the training images of a class on each pixel.
		initialize_index: int, optional
					Only need to specify when initialize = 'mean'. The index of a class on a known dataset.
		save: bool
				If true, the saliency map would be saved in the machine.

## Part 3: Visualization on self-built model of 5 class Imagenet Dataset

###visualize_softmax_dense
visualize_softmax_dense(model, iter, output_index = 0, step = 0.5, 
                             alpha = 0.1,save = True)
Parameters:
		model: 
				A keras training model.
		iter: int
				Number of times of iteration to run gradient ascent to maximize the loss function.
		output_index: int
				The index of a class (from 0 to 4) on 5-class Imagenet dataset.
		step: float
				The step of gradient ascent.
		alpha: float
				The regulization parameter of the loss function.
		save: bool
				If true, the saliency map would be saved in the machine.
				