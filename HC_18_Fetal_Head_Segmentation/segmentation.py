"""
Fetal head semantic segmentation main script
"""
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


from segmentation_utils import *

def load_image_train(sample_path):
	"""
	Load and preproces image for the segmentation

	Parameters
	----------
	image_file : string
		image's path

	Returns
	-------
	input_image : tensorflow tensor
		preprocessed CAM 
	real_image : tensorflow tensor
		preprocessed US image
	"""

	input_image = load_image(sample_path)

	## resize
	input_image = tf.image.resize(input_image, size=(192,272))

	## normalization
	mean = tf.math.reduce_mean(input_image)
	std = tf.math.reduce_std(input_image)
	input_image = (input_image - mean)/std
	

	return input_image

def load_mask_train(sample_path):
	"""
	Load and preproces the mask for the segmentation

	Parameters
	----------
	image_file : string
		image's path

	Returns
	-------
	input_image : tensorflow tensor
		preprocessed CAM 
	real_image : tensorflow tensor
		preprocessed US image
	"""

	input_image = load_image(sample_path)
	
	## resize
	input_image = tf.image.resize(input_image, size=(192,272))

	## normalization
	input_image = input_image / 255.
	
	return input_image

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for semantic segmentation of US fetal image')

	## images_path
	train_path = 'Dataset/training_set'


	## parameters dict
	semantic_dict = semantic_dict()
	
	img = load_image_train(train_path + '/' + '075_HC.png')
	mask = load_mask_train(train_path + '/' + '075_HC_Annotation.png')


	## MAKE tf.Dataset
	images_list, mask_list = image_mask_split(train_path)
	images_list = [train_path + '/' + i for i in images_list][:10]
	mask_list = [train_path + '/' + i for i in mask_list][:10]
	
	# NOTA: qui, giocando con image_list e mask list fai la divisione in train/val/test
	train_image_dataset = tf.data.Dataset.list_files(images_list, shuffle=False)
	train_image_dataset = train_image_dataset.map(load_image_train,
                                	  			num_parallel_calls=tf.data.AUTOTUNE)

	train_mask_dataset = tf.data.Dataset.list_files(mask_list, shuffle=False)
	train_mask_dataset = train_mask_dataset.map(load_mask_train,
                                	  			num_parallel_calls=tf.data.AUTOTUNE)
	## zip the images and the mask dataset
	train_dataset = tf.data.Dataset.zip((train_image_dataset, train_mask_dataset))
	train_dataset = train_dataset.batch(semantic_dict['batch_size'])
	# print(train_dataset)

	# for a, (image,mask) in enumerate(iter(train_dataset.take(5))):
	# 	fig, arr = plt.subplots(nrows=1, ncols=2, figsize=(12,6), num=f'US images and mask sample {a}', tight_layout=True)
	# 	arr[0].imshow(image, cmap='gray')
	# 	arr[0].set_title('US fetal head')
	# 	arr[0].axis('off')

	# 	arr[1].imshow(mask, cmap='gray')
	# 	arr[1].set_title('Segmentation mask')
	# 	arr[1].axis('off')
	# 	plt.show()


	## U-net model Training
	model = unet(input_size = (192,272,1))
	print(model.summary())
	
	model.compile(optimizer = Adam(learning_rate = semantic_dict['learning_rate']), 
	            loss = "binary_crossentropy", 
				metrics = ['accuracy'])

	history = model.fit(train_dataset, verbose = 1,
						batch_size = semantic_dict['batch_size'],
						epochs = semantic_dict['epochs'],
						)

		