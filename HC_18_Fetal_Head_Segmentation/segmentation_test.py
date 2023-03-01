"""
Test evaluation of segmentation model
"""
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import copy

from segmentation_utils import *
from segmentation import load_sample_test
from makedir import *
# from segmentation import make_dataset

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for semantic segmentation of US fetal image')
	parser.add_argument("-folder_name", default='trial', type=str, help="nome of folder to load the data")
	parser.add_argument("-model_epoch", default='20', type=str, help="number of model's epoch to load for testing")

	args = parser.parse_args()

	## images_path
	train_path = 'Dataset/training_set'
	save_folder = 'Model/' + args.folder_name

	## parameters dict
	semantic_dict = semantic_dict(epochs=20)
	
	## MAKE tf.Dataset train and validation
	training_path = 'Dataset/training_set_stack'
	sample_list = [training_path + '/' + i for i in os.listdir(training_path)]
	sample_list.sort()

	# split in train and test the whole dataset
	train_list, val_list, test_list = splitting_data(sample_list, splitting=(0.8,0.1,0.1))
	print(len(train_list), len(val_list), len(test_list))

	test_dataset = tf.data.Dataset.list_files(test_list[:4], shuffle=False)
	test_dataset = test_dataset.map(load_sample_test,
                                	  			num_parallel_calls=tf.data.AUTOTUNE)
	test_dataset = test_dataset.batch(1)
	
	## Prediction
	save_folder + '/weights/' + args.model_epoch + '.hdf5'
	model = tf.keras.models.load_model(save_folder + '/weights/' + args.model_epoch + '.hdf5')

	prediction = model.predict(test_dataset, verbose=1)
	pred = copy.copy(prediction)
	pred[pred > 0.5] = 1
	pred[pred < 0.5] = 0

	## save mask prediction
	results_path = save_folder + "/results"
	smart_makedir(results_path)
	
	for i in range(len(test_list[:4])):
		im = Image.fromarray((pred[i,:,:,0]* 255).astype(np.uint8))
		im.save(results_path + f"/pred_mask_{i}.png")

	## TEST vs VALIDATION curves
	accuracy = np.load(save_folder + '/history_accuracy.npy')
	val_accuracy = np.load(save_folder + '/history_val_accuracy.npy')
	loss = np.load(save_folder + '/history_loss.npy')
	val_loss = np.load(save_folder + '/history_val_loss.npy')

	## Dice coefficient
	for i,(image,mask) in enumerate(iter(test_dataset.take(4))):
		dice_coeff = dice(mask[0,:,:,:], pred[i,:,:,:])
		Dsc = DSC(mask[0,:,:,:], pred[i,:,:,:])
		print(f'dice: {dice_coeff}')
		print(f'DSC: {Dsc}')
		plt.figure()
		plt.imshow(mask[0,:,:,:], cmap='gray')

		plt.figure()
		plt.imshow(pred[i,:,:,:], cmap='gray')
		plt.show()

	## IoU
	# .....

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
	