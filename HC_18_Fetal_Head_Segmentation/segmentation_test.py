"""
Test evaluation of segmentation model
"""
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from segmentation_utils import *
from segmentation import make_dataset


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for semantic segmentation of US fetal image')

	## images_path
	train_path = 'Dataset/training_set'
	save_folder = 'Model'

	## parameters dict
	semantic_dict = semantic_dict(epochs=3)
	
	## MAKE tf.Dataset train and validation
	images_list, mask_list = image_mask_split(train_path)
	images_list = [train_path + '/' + i for i in images_list]
	mask_list = [train_path + '/' + i for i in mask_list]

	# split in train and test the whole dataset
	train_images_list, train_mask_list, test_images_list, test_mask_list = splitting_data(images_list, mask_list, tr=0.9)
	
	test_dataset = make_dataset(test_images_list, test_mask_list)
	test_dataset = test_dataset.batch(1)

	## Prediction
	model = tf.keras.models.load_model('Model/first_try')
	prediction = model.predict(test_dataset, verbose=1)
	print(prediction.shape)


	## TEST vs VALIDATION curves
	accuracy = np.load(save_folder + '/history_accuracy.npy')
	val_accuracy = np.load(save_folder + '/history_val_accuracy.npy')
	loss = np.load(save_folder + '/history_loss.npy')
	val_loss = np.load(save_folder + '/history_val_loss.npy')

	## Plots
	fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,8), num='Train_Val_curves')
	acc = [0.] + accuracy
	val_acc = [0.] + val_accuracy

	ax[0].plot(acc, label='Training Accuracy')
	ax[0].plot(val_acc, label=f'Validation Accuracy: last epoch {val_accuracy[-1]:.4f}')
	ax[0].legend(loc='lower right')
	ax[0].set_ylabel('Accuracy')
	ax[0].set_ylim([min(plt.ylim()),1])
	ax[0].set_title('Training and Validation Accuracy')

	ax[1].plot(loss, label='Training Loss')
	ax[1].plot(val_loss, label=f'Validation Loss: last epoch {val_loss[-1]:.4f}')
	ax[1].legend(loc='upper right')
	ax[1].set_ylabel('Loss')
	ax[1].set_ylim([0,0.6])
	ax[1].set_title('Training and Validation Loss')
	ax[1].set_xlabel('epoch')
	
	plt.figure()
	plt.imshow(prediction[0,:,:,:], cmap='gray')
	plt.show()