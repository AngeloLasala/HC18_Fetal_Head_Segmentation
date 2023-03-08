"""
Test evaluation of segmentation model
"""
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import copy
import random
from segmentation_utils import *
from segmentation import load_sample_test
from makedir import *
# from segmentation import make_dataset

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for semantic segmentation of US fetal image')
	parser.add_argument("-folder_name", default='trial', type=str, help="nome of folder to load the data")
	parser.add_argument("-model_epoch", default='20', type=str, help="number of model's epoch to load for testing")
	parser.add_argument('-save_img', action='store_true', help='save the mask prediction in selected folder. default=False')

	args = parser.parse_args()

	## images_path
	train_path = 'Dataset/training_set'
	save_folder = 'Model/' + args.folder_name

	## parameters dict
	semantic_dict = semantic_dict(epochs=30)
	
	## MAKE tf.Dataset train and validation
	training_path = 'Dataset/training_set_stack'
	sample_list = [training_path + '/' + i for i in os.listdir(training_path)]
	# sample_list.sort()  ## old version - sorted data
	random_index = np.load(save_folder + '/random_index.npy')
	sample_list = [sample_list[i] for i in random_index]
	for i,j in zip(random_index[:10], sample_list[:10]):
		print(i,j)
	
	## splitting
	train_list, val_list, test_list = splitting_data(sample_list, splitting=(0.8,0.1,0.1))
	for aa in train_list[:10]:
		print(aa)
	print(len(train_list), len(val_list), len(test_list))

	test_dataset = tf.data.Dataset.list_files(test_list, shuffle=False)
	test_dataset = test_dataset.map(load_sample_test,
                                	  			num_parallel_calls=tf.data.AUTOTUNE)
	test_dataset = test_dataset.batch(1)
	
	## Prediction
	model = tf.keras.models.load_model(save_folder + '/weights/' + args.model_epoch + '.hdf5')

	prediction = model.predict(test_dataset, verbose=1)
	pred = copy.copy(prediction)
	pred[pred > 0.5] = 1
	pred[pred < 0.5] = 0

	## save mask prediction
	results_path = save_folder + "/results"
	if args.save_img:
		smart_makedir(results_path)
		for i in range(len(test_list)):
			im = Image.fromarray((pred[i,:,:,0]*255).astype(np.uint8))
			name_pred = test_list[i].split('/')[-1]
			im.save(results_path + '/pred_' + name_pred)

			fig, arr = plt.subplots(nrows=1, ncols=3, figsize=(12,6), num=f'US images and mask sample{i}', tight_layout=True)

	## TEST vs VALIDATION curves
	accuracy = np.load(save_folder + '/history_accuracy.npy')
	val_accuracy = np.load(save_folder + '/history_val_accuracy.npy')
	loss = np.load(save_folder + '/history_loss.npy')
	val_loss = np.load(save_folder + '/history_val_loss.npy')

	## Dice coefficient
	DSC_tot = []
	hc_path = save_folder + "/hc_image"
	smart_makedir(hc_path)
	for i,(image,mask) in enumerate(iter(test_dataset.take(len(train_list)))):
		dice = DSC(mask[0,:,:,:], pred[i,:,:,:])
		print(f'sample {i}: dice = {dice}')

		# image = Image.fromarray((image[0,:,:,0]).numpy().astype(np.uint8))
		mask_s = Image.fromarray((mask[0,:,:,0]*255).numpy().astype(np.uint8))
		pred_mask_s = Image.fromarray((pred[i,:,:,0]*255).astype(np.uint8))

		# image.save(hc_path + f'/image_{i}.png')
		mask_s.save(hc_path + f'/mask_{i}.png')
		pred_mask_s.save(hc_path + f'/pred_mask_{i}.png')

		fig, arr = plt.subplots(nrows=1, ncols=3, figsize=(12,6), num=f'US images and mask sample{i}', tight_layout=True)
		arr[0].imshow(image[0,:,:,:], cmap='gray')
		arr[0].set_title('US fetal head')
		arr[0].axis('off')

		arr[1].imshow(mask[0,:,:,:], cmap='gray')
		# arr[1].imshow(ell_real, cmap='jet',alpha=0.6)
		arr[1].set_title('Segmentation mask')
		arr[1].axis('off')

		arr[2].imshow(pred[i,:,:,:], cmap='gray')
		# arr[2].imshow(ell, cmap='jet',alpha=0.6)
		arr[2].set_title('Prediction mask + fitting ellipse')
		arr[2].axis('off')
		plt.savefig(hc_path + f'/US images and mask sample{i}.png')
		plt.close()
		DSC_tot.append(dice)
		
	DSC_tot = np.array(DSC_tot)
	print(DSC_tot.shape)
	print(f'DSC: {DSC_tot.mean()} +- {DSC_tot.std()}')


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
	plt.savefig(save_folder + '/Train_Val_curves')
	# plt.show()
	