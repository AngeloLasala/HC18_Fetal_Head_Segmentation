"""
Utils module for Semantic Segmentation
"""
import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from typing_extensions import Concatenate
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint,  CSVLogger
from tensorflow.keras import backend as k
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomTranslation, RandomCrop

def semantic_dict(x = 192, 
          	     y = 272,
				batch_size = 8,
				n_channels_mask = 1,
				n_channels = 1,
				shuffle = True, 
				learning_rate = 0.0001,
				momentum = 0.98,
				epochs = 20):
	"""
	Dictionary of parameters for semantic segmentation
	"""
	sem_dict = {}

	# input shape and images feature
	sem_dict['x'] = x
	sem_dict['y'] = y
	sem_dict['n_channels_mask'] = n_channels_mask
	sem_dict['n_channels'] = n_channels

	# training parameters
	sem_dict['batch_size'] = batch_size
	sem_dict['shuffle'] = shuffle
	sem_dict['learning_rate'] = learning_rate
	sem_dict['momentum'] = momentum
	sem_dict['epochs'] = epochs

	return sem_dict

def load_image(sample_path):
		"""
		Load a SINGLE image from the input path

		Parameter
		---------
		main_path : string
			main path of folder

		sample_name : string
			image's path

		Returns
		------
		input_image : tensorflow tensor
			input imgage
	
		"""
		
		raw_image = tf.io.read_file(sample_path)
		image = tf.image.decode_png(raw_image, channels=1)

		input_image = tf.cast(image, tf.float32)
		
		return input_image

def image_mask_split(path):
	"""
	Visualize the US images and the relative mask

	Parameters
	----------
	path : string
		path of training dataset
	
	Returns
	-------
	images_list : list
		list of images' path

	mask_list: list
		list of masks' path
	"""

	print(len(os.listdir(path)))
	images_list, mask_list = [], []

	for img_path in os.listdir(path):
		#take the name (delate .png)
		img_path_plane = img_path.split('.')[0]

		#take the real and mask
		if len(img_path_plane.split('_')) == 2: images_list.append(img_path)
		if len(img_path_plane.split('_')) == 3: mask_list.append(img_path)

	images_list.sort()
	mask_list.sort()

	return images_list, mask_list

def random_shuffle(images_list):
	"""
	The sorted list of data have a problem, the head circunferance are arrange 
	in a crescent fashion. This must be a problem for the generalization and for
	the hc measure.

	The data are randomly shuffle before the splitting

	Parameters
	----------
	images_list : list
		original list of sample

	Returns
	-------
	"""
	## random shuffle of the images list indeces
	random_list = np.arange(len(images_list))
	np.random.shuffle(random_list)
	
	## shuffle the image list
	images_list = [images_list[i] for i in random_list]

	return random_list, images_list




def splitting_data(images_list, splitting=(0.9,0.1,0.1)):
	"""
	Return the list of data for training, test. Note that for the validation
	there is the other function built for the cross validation

	Parameters
	----------
	image_list : list
		list of image

	mask_list : list
		mask of image

	Returns
	-------
	train_images_list : list
		train image list
	
	train_mask_list : list
		train mask list 

	test_images_list : list
		test image list
	
	test_mask_list : list
		test mask list

	"""	
	## compute the index for splitting
	tot = len(images_list)
	train_ind = int(tot*splitting[0])
	val_index = int(tot*splitting[1])
	test_index = int(tot*splitting[2])


	## split using the list index
	train_list = images_list[:train_ind]
	val_list = images_list[train_ind:train_ind+val_index]
	test_list = images_list[train_ind+val_index:]

	return train_list, val_list, test_list

#######################################################################################################

def data_augmenter():
	"""
	Create a Sequential model composed of 4 layers

	Returns
	------- 
	data_augumentation: tf.keras.Sequential

	"""
	data_augmentation = tf.keras.Sequential()
	data_augmentation.add(RandomFlip())
	data_augmentation.add(RandomRotation(0.2)) # 15 degrees
	# data_augmentation.add(RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05))) # 10 pixel
	# data_augmentation.add(RandomCrop(224-int(0.1*224),224-int(0.1*224)))
	
	return data_augmentation

def resize(input_image, mask, height, width):
	"""
	Resize the input and the real image for gan

	Parameters
	----------
	input_image : tensorflow tensor
		input imgage, i.e. real US image 

	mask : tensorflow tensor
		seg mask

	height : integer
		height of resized image 

	width : integer
		width of resized image

	Returns
	-------
	input_image : tensorflow tensor
		resized CAM 

	real_image : tensorflow tensor
		resized US image
	"""
	input_image = tf.image.resize(input_image, [height, width],
								method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	mask = tf.image.resize(mask, [height, width],
								method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	return input_image, mask

def random_crop(input_image, mask, height, width):
	"""
	Random cropping of input and the real image for gan

	Parameters
	----------
	input_image : tensorflow tensor
		input imgage, i.e. CAM 

	real_image : tensorflow tensor
		real image, i.e. US image

	height : integer
		height of cropped image 

	width : integer
		width of cropped image
	Returns
	-------
	cropped_image : tensorflow tensor
		cropped CAM 

	cropped_image : tensorflow tensor
		cropped US image
	"""
	stacked_image = tf.stack([input_image, mask], axis=0)
	cropped_image = tf.image.random_crop(
		stacked_image, size=[2, height, width, 1])

	return cropped_image[0], cropped_image[1]
	    

def unet(input_size = (192,272,1)): #(params['x'],params['y'],1)
    inputs = Input(input_size)
    
    ## 1 block
    conv1 = Conv2D(64,3, padding='same', activation='relu')(inputs)
    conv2 = Conv2D(64,3, padding='same', activation='relu')(conv1)
    pool1 = MaxPool2D()(conv2)

    ## 2 block
    conv3 = Conv2D(128,3, padding='same', activation='relu')(pool1)
    conv4 = Conv2D(128,3, padding='same', activation='relu')(conv3)
    pool2 = MaxPool2D()(conv4)

    ## 3 block
    conv5 = Conv2D(256,3, padding='same', activation='relu')(pool2)
    conv6 = Conv2D(256,3, padding='same', activation='relu')(conv5)
    pool3 = MaxPool2D()(conv6)

    ## 4 block
    conv7 = Conv2D(512,3, padding='same', activation='relu')(pool3)
    conv8 = Conv2D(512,3, padding='same', activation='relu')(conv7)
    pool4 = MaxPool2D()(conv8)

    ## Bottleneck
    conv9 = Conv2D(1024,3, padding='same', activation='relu')(pool4)
    conv10 = Conv2D(1024,3, padding='same', activation='relu')(conv9)

    ## 1 up-block
    up1 =  UpSampling2D()(conv10)
    conv11 = Conv2D(512,2, padding='same', activation='relu')(up1)
    conc1 = Concatenate()([conv11,conv8])
    conv12 = Conv2D(512,3, padding='same', activation='relu')(conc1)
    conv13 = Conv2D(512,3, padding='same', activation='relu')(conv12)

    ## 2 up-block
    up2 =  UpSampling2D()(conv13)
    conv14 = Conv2D(256,2, padding='same', activation='relu')(up2)
    conc2 = Concatenate()([conv14,conv6])
    conv15 = Conv2D(256,3, padding='same', activation='relu')(conc2)
    conv16 = Conv2D(256,3, padding='same', activation='relu')(conv15)

    ## 3 up-block
    up3 =  UpSampling2D()(conv16)
    conv17 = Conv2D(128,2, padding='same', activation='relu')(up3)
    conc3 = Concatenate()([conv17,conv4])
    conv18 = Conv2D(128,3, padding='same', activation='relu')(conc3)
    conv19 = Conv2D(128,3, padding='same', activation='relu')(conv18)

    ## 4 up-block
    up4 =  UpSampling2D()(conv19)
    conv20 = Conv2D(64,2, padding='same', activation='relu')(up4)
    conc4 = Concatenate()([conv20,conv2])
    conv21 = Conv2D(64,3, padding='same', activation='relu')(conc4)
    conv22 = Conv2D(64,3, padding='same', activation='relu')(conv21)

    ## last output
    conv23 = Conv2D(2,3, padding='same', activation='relu')(conv22)
    output = Conv2D(1,1, padding='same', activation='sigmoid')(conv23)

    #### to fill 
    model = Model(inputs,output)

    return model

## U_net VGG16 #######################################################################################
def conv_block(input, num_filters):
    """
    Convolutional unit of VGG 16
    """
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
 
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
 
    return x
 
def decoder_block(input, skip_features, num_filters):
    """
    Transpose convolution block of VGG 16 with skip connection
    """
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x
 
def vgg16_unet(input_shape, weight='imagenet', trainable = False):
    """
    Unet architecture based on VGG 16 backbnone. Note that the last 
    activation is tanh as in the generator of pix2pix

    Parameters
    ----------
    input_shape : tuple
        input shape

    weight : string
        download the weights of the VGG16 model:
        - 'imagenet': load the VGG16 model pretrained on imagenet
        - 'path/to/model' (i.e.'Images_classification_Brain_plane/models/VGG_16_/train_11/VGG_16'):
           load the retrained model on Brain Plane dataset

        default = 'imagenet'

    trainable : bool
        retraine the parameter of the VGG16 encoder. default='False'

    Returns
    -------
    model : tensorflow model
        VGG16 U net
    """

    ## input layer
    inputs = tf.keras.layers.Input(input_shape)  ## the comment refers to a input shape (256,256,3)
 
    ## pretrained model of vgg16
    if weight == 'imagenet':
        vgg16 = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    
    if weight == 'random':
        vgg16 = tf.keras.applications.VGG16(include_top=False, weights=None, input_tensor=inputs)
	
    vgg16.trainable = trainable #retrain or not the encoder leyer  
 
    ## encoder
    s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output         ## (64 x 64)
 
    ## bottleneck
    b1 = vgg16.get_layer("block5_conv3").output         ## (32 x 32)
 
    ## decoder with skip connection
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)
 
    ## last layer 
    initializer = tf.random_normal_initializer(0., 0.02)
    outputs = tf.keras.layers.Conv2D(3, 3, padding="same", kernel_initializer=initializer, activation="tanh")(d4)   ## activation is tanh for as in pix2pix generator model
 
    model = tf.keras.Model(inputs, outputs, name="VGG16_U-Net")
    return model
#####################################################################################################

def dataset_visualization(dataset, take_ind=5):
	"""
	Plot data from dataset object
	"""
	for a, (image,mask) in enumerate(iter(dataset.take(take_ind))):
		fig, arr = plt.subplots(nrows=1, ncols=2, figsize=(12,6), num=f'US images and mask sample {a}', tight_layout=True)
		arr[0].imshow(image[0,:,:,:], cmap='gray')
		arr[0].set_title('US fetal head')
		arr[0].axis('off')

		arr[1].imshow(mask[0,:,:,:], cmap='gray')
		arr[1].set_title('Segmentation mask')
		arr[1].axis('off')
	
#####################################################################################################################################ù

def dice(im1, im2, empty_score=1.0):
	"""
	Computes the Dice coefficient, a measure of set similarity.
	Parameters
	----------
	im1 : array-like, bool
		Any array of arbitrary size. If not boolean, will be converted.
	im2 : array-like, bool
		Any other array of identical size. If not boolean, will be converted.
	Returns
	-------
	dice : float
		Dice coefficient as a float on range [0,1].
		Maximum similarity = 1
		No similarity = 0
		Both are empty (sum eq to zero) = empty_score
		
	Notes
	-----
	The order of inputs for `dice` is irrelevant. The result will be
	identical if `im1` and `im2` are switched.
	"""
	im1 = np.asarray(im1).astype(bool)
	im2 = np.asarray(im2).astype(bool)
	
	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

	if len(im1.shape)==2:
		im_sum = im1.sum() + im2.sum()
		# im_sum = np.logical_or(im1, im2).sum()
		if im_sum == 0:
			return empty_score

		# Compute Dice coefficient
		intersection = np.logical_and(im1, im2).sum()

		print("intersection:", intersection)
		print("union:", im_sum)

		return 2. * intersection.sum() / im_sum.sum()

	else:
		im_sum = 0
		intersection = 0
		for c in range(im1.shape[-1]):
			im_sum += im1[:,:,c].sum() + im2[:,:,c].sum()
			# im_sum = np.logical_or(im1[0,:,:,c], im2[0,:,:,c]).sum()
			if im_sum == 0:
				return empty_score

			# Compute Dice coefficient
			intersection = np.logical_and(im1[:,:,c], im2[:,:,c]).sum()

			print("intersection on channel" + str(c) + ": " + str(intersection))
			print("union on channel" + str(c) + ": " + str(im_sum))

			return 2. * intersection.sum() / im_sum.sum()

def DSC(im1, im2):
	"""
	Dice similarity coefficient for single channel images

	Parameters
	----------
	im1: 2darray
		first image

	im2: 2darray
		second image

	Results
	-------
	dice: float
		dice similarity coefficient
	"""
	im1 = np.asarray(im1).astype(bool)
	im2 = np.asarray(im2).astype(bool)

	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

	TP = np.logical_and(im1,im2).sum()    
	FP_FN = np.logical_xor(im1,im2).sum() 

	# print("true positives:", TP)
	# print("false neg + false pos:", FP_FN)

	dsc = 2.*TP/(2.*TP+FP_FN)

	return dsc