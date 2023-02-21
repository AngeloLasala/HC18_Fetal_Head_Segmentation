"""
Simple script to visualize the images: US + mask
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt 

def visualize_us_mask(path):
	"""
	Visualize the US images and the relative mask

	Parameters
	----------
	path : string
		path of training dataset
	
	Returns
	-------
	"""

	print(os.listdir(path))

if __name__ == '__main__':

	train_path = '/Dataset/training_set'

	visualize_us_mask(train_path)

