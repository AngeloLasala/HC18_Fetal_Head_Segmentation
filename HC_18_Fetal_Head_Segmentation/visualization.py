"""
Simple script to visualize the images: US + mask
"""
import argparse
import os
from PIL import Image
import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt 

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

	# print(len(os.listdir(path)))
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

def fill_mask(mask_list, training_path='Dataset/training_set_original', N=0):
	"""
	Fill the inside of the mask to resolve the issue about the anbalanced of data

	Parameters
	----------
	mask_path : string 
		path of the mask


	Returns
	-------
	"""
	imgray = cv.imread(training_path + '/' + mask_list[N], cv.IMREAD_GRAYSCALE)
	contours, hierarchy = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	filled_image = np.zeros_like(imgray)
	filled_img = cv.fillPoly(filled_image, contours, color=255)
	return filled_img 

def save_filled_mask(mask_list, training_path='Dataset/training_set'):
	"""
	Save the filled mask

	Parameters
	----------
	mask_list: list
		list of masks' path

	training_path : string
		traing path
	"""
	for N in range(len(mask_list)):
		print(f'filling image {N}')
		mask = fill_mask(mask_list, training_path='Dataset/training_set', N=N)
		cv.imwrite(training_path + '/' + mask_list[N], mask)

def saving_stack_image(images_list, mask_list, dim=(224,224), training_path_save='Dataset/training_set_stack'):
	"""
	Saving the image in vertical stack fascion to make easy the dataaugumentation
	"""
	for N in range(len(mask_list)):
		mask = fill_mask(mask_list, training_path='Dataset/training_set_original', N=N)
		image = cv.imread(train_path + '/' + images_list[N], cv.IMREAD_GRAYSCALE)

		mask = cv.resize(mask, dim, interpolation=cv.INTER_CUBIC)
		image = cv.resize(image, dim, interpolation=cv.INTER_CUBIC)
		print(N, mask.shape, image.shape)

		stack_image = np.vstack((mask,image))
		plt.figure()
		plt.imshow(stack_image, cmap='gray')
		
		cv.imwrite(training_path_save + '/' + mask_list[N], stack_image)
		plt.close()
		
def visualize_img_mask(image_list, mask_list, training_path='Dataset/training_set_original', N=6, filling=False):
	"""
	Visualize the image and the relative mask

	Parameters
	----------
	images_list : list
		list of images' path

	mask_list: list
		list of masks' path

	training_path : string
		traing path

	N : integer
		position of the images in the list 
	
	filling : bool
		if True, the cirvle/ellipse is filled
	"""
	img = cv.imread(training_path + '/' + image_list[N])
	mask = cv.imread(training_path + '/' + mask_list[N])

	if filling:
		mask = fill_mask(mask_list, training_path='Dataset/training_set', N=N)

	fig, arr = plt.subplots(nrows=1, ncols=2, figsize=(12,6), num=f'US images and mask sample {N} {filling}', tight_layout=True)
	arr[0].imshow(img, cmap='gray')
	arr[0].set_title('US fetal head')
	arr[0].axis('off')

	arr[1].imshow(mask, cmap='gray')
	arr[1].set_title('Segmentation mask')
	arr[1].axis('off')

	return img, mask

if __name__ == '__main__':

	train_path = 'Dataset/training_set_original'
	N = 976

	## Split the images and the mask 
	images_list, mask_list = image_mask_split(train_path)

	## Visualize a couple of img and mask
	img, mask = visualize_img_mask(images_list, mask_list, N=N, filling=False)
	img, mask = visualize_img_mask(images_list, mask_list, N=N, filling=True)
		
	path = 'Dataset/training_set_original/787_HC_Annotation.png'
	mask_in = cv.imread(path, cv.IMREAD_GRAYSCALE)

	# for i in range(364,429):
	# 	mask_in[1,i] = 255
	# 	print(mask_in[1,i])
	
	# print(mask_in[1,:])
	
	# cv.imwrite(path, mask_in)


	## Save filled mask
	saving = False
	if saving : save_filled_mask(mask_list)

	saving_stack = True
	if  saving_stack: 
		saving_stack_image(images_list, mask_list, dim=(224,224))

	# fill_mask(mask_list, training_path='Dataset/training_set', N=976)

	plt.show()