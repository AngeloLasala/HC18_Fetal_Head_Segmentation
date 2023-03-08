"""
From the segmentation output fit the ellipses to find the head circunferance
"""
import argparse
import os
from PIL import Image
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from visualization import visualize_img_mask
from makedir import *

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

def load_image_cv(image_path, thickness=1):
	"""
	Load image and mask using opencv4
	"""
	image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

	w = image.shape[0]
	w = w // 2
	mask = image[:w, :]
	real_image = image[w:, :]

	return real_image, mask

def fit_ellipse(input_image, thickness=1):
	"""
	Given an image (the mask or pred_mask) and return the
	elipse fitted on the countourn

	Parameters
	----------
	input_image : 2d array
		input image

	Returns
	-------
	ellipses : list
		ellipse parameters

	ell : array
		ellipse's image
	"""
	# load the images
	# mask_pred = cv.imread(input_image, cv.IMREAD_GRAYSCALE)

	# Find the contours in the mask
	contours, hierarchy = cv.findContours(input_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	# Fit an ellipse to each contour
	ellipses = []
	for cnt in contours:
		if len(cnt) >= 5:
			ellipse = cv.fitEllipse(cnt)
			ellipses.append(ellipse)

	# Draw the ellipses on a copy of the mask
	mask_with_ellipses = np.zeros_like(input_image)
	for ellipse in ellipses[:1]:
		ell = cv.ellipse(mask_with_ellipses, ellipse, (255), thickness=thickness)

	return ellipses, ell

def head_circunferance(ell_parameteres):
	"""
	Compute the head circunference of given ellipse parameters.
	Note: the results is given in pixels !!!

	Parameters
	----------
	ell_parameters : tuple
		the 5 parameters that characterize an ellipse

	Return
	------
	hc : float
		the head circunferenc in pixel
	"""

	a_axis = ell_parameteres[0][1][0]/2
	b_axis = ell_parameteres[0][1][1]/2

	hc = 2 * np.pi * np.sqrt((a_axis**2 + b_axis**2)/2)

	return hc

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for semantic segmentation of US fetal image')
	parser.add_argument("-folder_name", default='trial', type=str, help="nome of folder to load the data")
	parser.add_argument('-save_img', action='store_true', help='save the mask prediction in selected folder. default=False')
	args = parser.parse_args()

	save_folder = 'Model/' + args.folder_name
	training_path = 'Dataset/training_set_stack'

	## MAKE tf.Dataset train and validation
	# sample_list = [training_path + '/' + i for i in os.listdir(training_path)]
	# # sample_list.sort()  ## old version - sorted data
	# random_index = np.load(save_folder + '/random_index.npy')
	# sample_list = [sample_list[i] for i in random_index]
	# for i,j in zip(random_index[:10], sample_list[:10]):
	# 	print(i,j)
	
	# ## splitting
	# train_list, val_list, test_list = splitting_data(sample_list, splitting=(0.8,0.1,0.1))
	# for aa in train_list[:10]:
	# 	print(aa)
	# print(len(train_list), len(val_list), len(test_list))

	## Here I load the image of predicted mask correctly saved with original name
	# for the predicted mask
	hc_images_path = save_folder + "/hc_image"
	hc_save_folder = save_folder + "/hc_image_fit"

	real_mask_list, pred_mask_list = [], []
	for i in os.listdir(hc_images_path):
		if i.split('_')[0]=='mask':real_mask_list.append(i)
		if i.split('_')[0]=='pred':pred_mask_list.append(i)

	real_mask_list.sort()
	pred_mask_list.sort()

	# Fitting ellipse and compute the MAE
	average_err, hc_pred_list = [], []
	hc_pred_resize_list, average_resize_err = [], []
	smart_makedir(hc_save_folder)
	for i, (real, predd) in enumerate(zip(real_mask_list, pred_mask_list)):

		## Fit ellipses
		mask = cv.imread(hc_images_path  + '/' + real, cv.IMREAD_GRAYSCALE)
		mask_pred = cv.imread(hc_images_path  + '/' + predd, cv.IMREAD_GRAYSCALE)
		ellipse_pred, ell = fit_ellipse(mask_pred)
		ellipse_real, ell_real = fit_ellipse(mask)

		hc_pred = head_circunferance(ellipse_pred)
		hc_real = head_circunferance(ellipse_real)
		# print(f'hc_real:{hc_real:.4f} - hc_pred:{hc_pred:.4f} - err:{np.abs(hc_pred-hc_real)}')

		hc_pred_list.append(hc_pred)
		average_err.append(np.abs(hc_pred-hc_real))

		
		# resize to 540, 800
		mask_resize = cv.resize(mask, (800,540), interpolation=cv.INTER_CUBIC)
		mask_pred_resize = cv.resize(mask_pred, (800,540), interpolation=cv.INTER_CUBIC)

		ellipse_pred_resize, ell_resize = fit_ellipse(mask_pred_resize)
		ellipse_real_resize, ell_real_resize = fit_ellipse(mask_resize)

		hc_pred_resize = head_circunferance(ellipse_pred_resize)
		hc_real_resize = head_circunferance(ellipse_real_resize)

		fig, arr = plt.subplots(nrows=1, ncols=2, figsize=(10,6), num=f'mask and fitting of test sample {i}', tight_layout=True)
		
		arr[0].imshow(mask, cmap='gray')
		arr[0].imshow(ell_real, cmap='jet',alpha=0.6)
		arr[0].set_title('Segmentation mask')
		arr[0].axis('off')

		arr[1].imshow(mask_pred, cmap='gray')
		arr[1].imshow(ell, cmap='jet',alpha=0.6)
		arr[1].set_title('Prediction mask + fitting ellipse')
		arr[1].axis('off')
		plt.savefig(hc_save_folder + '/' + f'mask and fitting of test sample {i}')
		# plt.show()
		plt.close()
		print(f'hc_real:{hc_real:.4f} - hc_pred:{hc_pred:.4f} - err:{np.abs(hc_pred-hc_real)}')

		hc_pred_resize_list.append(hc_pred_resize)
		average_resize_err.append(np.abs(hc_pred_resize-hc_real_resize))
		


	average_err, hc_pred_list = np.array(average_err), np.array(hc_pred_list)
	HC_pred, MEA = np.mean(hc_pred_list), np.mean(average_err)
	print(f'HC_pred: {HC_pred:2f}, MAE: {MEA:.2f} (px)')

	average_resize_err, hc_pred_resize_list = np.array(average_resize_err), np.array(hc_pred_resize_list)
	HC_pred, MEA = np.mean(hc_pred_resize_list), np.mean(average_resize_err)
	print(f'HC_pred: {HC_pred:2f}, MAE: {MEA:.2f} (px - resize)')

	#Rough estimation of MAE in mm
	pd_train = pd.read_csv('Dataset/training_set_pixel_size_and_HC.csv')
	mm_per_pixel = pd_train['pixel size(mm)'].mean() #mean value accros all the samples

	average_resize_err, hc_pred_resize_list = np.array(average_resize_err)*mm_per_pixel, np.array(hc_pred_resize_list)*mm_per_pixel
	HC_pred, MEA = np.mean(hc_pred_resize_list), np.mean(average_resize_err)
	print(f'HC_pred: {HC_pred:2f}, MAE: {MEA:.2f} (mm - resize)')

	### OLD VERSION ###############################################################################
	# ## MAE in millimenters
	# pd_train = pd.read_csv('Dataset/training_set_pixel_size_and_HC.csv')
	# fitting_path = save_folder + '/fitting'
	# smart_makedir(fitting_path)
	# average_err, hc_pred_list = [], []
	# for i in range(len(test_list[:2])):
		
	# 	img, mask = load_image_cv(test_list[i])
	# 	results_path = save_folder + "/results"

	# 	# take the value of mm per pxl for each image
	# 	mm_per_pixel = pd_train.iloc[i+len(train_list)+len(val_list)]['pixel size(mm)']

	# 	## Fit ellipses
	# 	mask_pred = cv.imread(results_path + f'/pred_mask_{i}.png', cv.IMREAD_GRAYSCALE)
		
	# 	#resize the mask to original shape
	# 	img = cv.resize(img, (800,540), interpolation=cv.INTER_CUBIC)
	# 	mask = cv.resize(mask, (800,540), interpolation=cv.INTER_CUBIC)
	# 	mask_pred = cv.resize(mask_pred, (800,540), interpolation=cv.INTER_CUBIC)

	# 	#fit ellipse
	# 	ellipse_pred, ell = fit_ellipse(mask_pred)
	# 	ellipse_real, ell_real = fit_ellipse(mask)
	
	# 	hc_pred = head_circunferance(ellipse_pred) * mm_per_pixel
	# 	hc_real = head_circunferance(ellipse_real) * mm_per_pixel
	# 	print(f'{i+len(train_list)+len(val_list)}) hc_real:{hc_real:.4f} - hc_pred:{hc_pred:.4f} - err:{np.abs(hc_pred-hc_real)}')

	# 	fig, arr = plt.subplots(nrows=1, ncols=3, figsize=(12,6), num=f'US images and mask sample {i}', tight_layout=True)
	# 	arr[0].imshow(img, cmap='gray')
	# 	arr[0].set_title('US fetal head')
	# 	arr[0].axis('off')

	# 	arr[1].imshow(mask, cmap='gray')
	# 	arr[1].imshow(ell_real, cmap='jet',alpha=0.6)
	# 	arr[1].set_title('Segmentation mask')
	# 	arr[1].axis('off')

	# 	arr[2].imshow(mask_pred, cmap='gray')
	# 	arr[2].imshow(ell, cmap='jet',alpha=0.6)
	# 	arr[2].set_title('Prediction mask + fitting ellipse')
	# 	arr[2].axis('off')
	# 	plt.savefig(fitting_path + f'/US images and mask sample {i}')
	# 	plt.close()

	# 	hc_pred_list.append(hc_pred)
	# 	average_err.append(np.abs(hc_pred-hc_real))
		
	# average_err, hc_pred_list = np.array(average_err), np.array(hc_pred_list)
	# HC_pred, MEA = np.mean(hc_pred_list), np.mean(average_err)
	# print(f'HC_pred: {HC_pred:2f} (mm), MAE: {MEA:.2f} (mm)')
	# print(img.shape, ell_real.shape)

	# fig, arr = plt.subplots(nrows=1, ncols=3, figsize=(12,6), num=f'US images and mask sample {i}', tight_layout=True)
	# arr[0].imshow(img, cmap='gray')
	# arr[0].set_title('US fetal head')
	# arr[0].axis('off')

	# arr[1].imshow(mask, cmap='gray')
	# arr[1].imshow(ell_real, cmap='jet',alpha=0.6)
	# arr[1].set_title('Segmentation mask')
	# arr[1].axis('off')

	# arr[2].imshow(mask_pred, cmap='gray')
	# arr[2].imshow(ell, cmap='jet',alpha=0.6)
	# arr[2].set_title('Prediction mask + fitting ellipse')
	# arr[2].axis('off')

	plt.show()



	