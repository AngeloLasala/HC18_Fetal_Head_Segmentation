"""
Utils module for Semantic Segmentation
"""
import os
import numpy as np
import tensorflow as tf 
from typing_extensions import Concatenate
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint,  CSVLogger
from tensorflow.keras import backend as k
from tensorflow.keras.initializers import glorot_uniform

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


# seed = 42
# train_datagen = ImageDataGenerator(rotation_range=25, fill_mode='constant')
# train_mask_datagen = ImageDataGenerator(rotation_range=25, fill_mode='constant')

# train_datagen.fit(X_train, augment=True, seed=seed)
# train_mask_datagen.fit(Y_train, augment=True, seed=seed)
# train_image_generator = train_datagen.flow(X_train,batch_size=params['batch_size'], 
# 											shuffle=True,seed=seed)
# train_mask_generator = train_mask_datagen.flow(Y_train,batch_size=params['batch_size'], 
# 											   shuffle=True, seed=seed)

# train_generator = zip(train_image_generator,train_mask_generator)

#######################################################################################################

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