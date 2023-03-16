"""
Apply the segmentation model on the sample for GAN
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import copy

from segmentation import normalize
from makedir import *

def load_image(sample_path):
		"""
		Load a SINGLE couple of image and mask for segmentation
		Parameter
		---------
		main_path : string
			main path of folder

		sample_name : string
			image's path

		Returns
		------
		real_image : tensorflow tensor
			real image, i.e. US image

		mask : tensorflow tensor
			mask for segmentation
		"""
		
		raw_image = tf.io.read_file(sample_path)
		image = tf.image.decode_png(raw_image, channels=1)

		w = tf.shape(image)[0]
		w = w // 2
		mask = image[:w, :, :]
		real_image = image[w:, :, :]

		mask = tf.cast(mask, tf.float32)
		input_image = tf.cast(real_image, tf.float32)

		mask = tf.image.resize(mask, (224,224))
		input_image = tf.image.resize(input_image, (224,224))
		
		return input_image, mask

def load_sample_test(sample_path):
	"""
	Load and preproces train_file
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
	input_image, mask = load_image(sample_path)
	input_image, mask = normalize(input_image, mask)

	return input_image, mask

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for semantic segmentation of US fetal image')
	parser.add_argument("-folder_name", default='trial', type=str, help="nome of folder to load the data")
	parser.add_argument("-model_epoch", default='20', type=str, help="number of model's epoch to load for testing")

	args = parser.parse_args()

	## images_path
	gan_path = 'Dataset/train_TC'
	save_folder = 'Model/' + args.folder_name
	
	gan_list = [gan_path + '/test' + '/' + i for i in os.listdir(gan_path + '/test')]
	gan_list.sort() ## IMPORTANT for the enumeration 
	
	gan_dataset = tf.data.Dataset.list_files(gan_list, shuffle=False)
	gan_dataset = gan_dataset.map(load_sample_test,
                                	  			num_parallel_calls=tf.data.AUTOTUNE)
	gan_dataset = gan_dataset.batch(1)

	## Prediction
	model = tf.keras.models.load_model(save_folder + '/weights/' + args.model_epoch + '.hdf5')

	prediction = model.predict(gan_dataset, verbose=1)
	pred = copy.copy(prediction)
	pred[pred > 0.5] = 1
	pred[pred < 0.5] = 0


	gan_result = gan_path + '/segmentation_test'
	raw_segmentation_path = gan_path + '/raw_segmentation_test'
	mask_path = gan_path + '/mask_test'
	smart_makedir(gan_result)
	smart_makedir(raw_segmentation_path)
	smart_makedir(mask_path)
	for i,(image,cam) in enumerate(iter(gan_dataset.take(len(gan_list)))):
		sample_num = gan_list[i].split('/')[-1].split('.')[0].split('_')[-1]
		print(sample_num)
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6), num=f'US image and mark, gan {sample_num}')
		ax[0].imshow(image[0,:,:,:], cmap='gray')
		ax[0].set_title('US fetal head')
		ax[0].axis('off')

		ax[1].imshow(pred[i,:,:,:], cmap='gray')
		ax[1].set_title('Segmentation mask')
		ax[1].axis('off')
		plt.savefig(gan_result + '/' + f'US image and mark, gan {sample_num}')
		plt.close()

		fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,6), num=f'Raw Segmentation {sample_num}')
		ax1.imshow(image[0,:,:,:], cmap='gray')
		ax1.imshow(pred[i,:,:,:], cmap='jet', alpha = 0.5)
		ax1.axis('off')
		plt.savefig(raw_segmentation_path + '/' + f'Raw Segmentation {sample_num}')
		plt.close()

		#save only mask for ellipses fitting
		im = Image.fromarray((pred[i,:,:,0]*255).astype(np.uint8))
		im.save(mask_path + f'/mask_{sample_num}.png')
		
		plt.show()

