"""
Build the dataset for pix2pix GAN with Elliptic loss
Input=CAM   target=US_image  shape=segmentation_mask  
"""
import argparse
import os
from PIL import Image
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from makedir import *
from hc_fitting import fit_ellipse

def load_image_cv(sample_path):
		"""
		Load a SINGLE couple of input_image and real_image for GAN
		Parameter
		---------
		main_path : string
			main path of folder
		sample_name : string
			image's path
		Returns
		------
		input_image : tensorflow tensor
			input imgage, i.e. CAM 
		real_image : tensorflow tensor
			real image, i.e. US image
		"""
		
		image = cv.imread(sample_path, cv.IMREAD_COLOR)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

		w = image.shape[0]
		w = w // 2
		input_image = image[:w, :, :]
		real_image = image[w:, :, :]

		return input_image, real_image

def process_mask(mask_path, dim):
	"""
	load the mask and preprocess it

	Parameters
	----------
	mask_path: string
		path of the mask
	
	dim : array
		final dimensio of the mask

	Returns
	-------
	mark: array
		processed mask
	"""
	mask = cv.imread(m_path, cv.IMREAD_GRAYSCALE)

	# resize
	mask = cv.resize(mask, dim, interpolation = cv.INTER_AREA)

	# normalize
	# mask = mask / 255.

	return mask


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='make the data sample for pix2pix train (cam,image,mask)')
	parser.add_argument("fetal_plane", default='', type=str, help="nome of featl standard planes (TC, TT, TV)")
	parser.add_argument("-train_or_test", default='train', type=str, help="select train or test dataset")
	parser.add_argument('-save_img', action='store_true', help='save the mask prediction in selected folder. default=False')
	args = parser.parse_args()

	# folder path
	gan_folder = 'Dataset/train_' + args.fetal_plane 
	gan_samples_folder = gan_folder + '/gan_images_' + args.train_or_test
	mask_folder = gan_folder + '/mask_' + args.train_or_test
	gan_segm = 'Dataset/train_' + args.fetal_plane + '/' + args.train_or_test + '_' + args.fetal_plane + '_stack'

	samples_path = [gan_samples_folder + '/' + i for i in os.listdir(gan_samples_folder)]
	mask_path = [mask_folder + '/' + i for i in os.listdir(mask_folder)]
	samples_path.sort()
	mask_path.sort()

	smart_makedir(gan_segm)
	for (s_path, m_path) in zip(samples_path,mask_path):
		sample_num = s_path.split('/')[-1].split('.')[0].split('_')[-1]

		## load sample and process the mask
		cam, image = load_image_cv(s_path)
		mask = process_mask(mask_path, dim=(image.shape[1], image.shape[0]))
		print(f'resolution: {image.shape}')

		ellipses_par, ell_mask = fit_ellipse(mask, thickness=-1)
		print(ellipses_par)

		fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(4,12), num=f'sample_gan_{sample_num}')
		ax[0].imshow(cam, cmap='jet')
		ax[0].set_title('CAM')
		ax[0].axis('off')

		ax[1].imshow(image, cmap='gray')
		ax[1].imshow(mask, cmap='jet', alpha=0.3)
		ax[1].set_title('US IMAGE + SEGMENTATION')
		ax[1].axis('off')

		ax[2].imshow(image, cmap='gray')
		ax[2].imshow(ell_mask, cmap='jet', alpha=0.3)
		ax[2].set_title('US IMAGE + ELLIPTIC SEGMENTATION')
		ax[2].axis('off')
		# plt.savefig(gan_result + '/' + f'US image and mark, gan {sample_num}')
		plt.close()

		## stank (cam, image, mask)
		mask = cv.cvtColor(ell_mask, cv.COLOR_GRAY2RGB)
		stack_sample = np.vstack((cam, image, mask))

		stack_img = Image.fromarray((stack_sample.astype(np.uint8)))
		stack_img.save(gan_segm + f'/sample_{sample_num}.png')
		
		# plt.figure()
		# plt.imshow(stack_sample)
		# plt.show()


	