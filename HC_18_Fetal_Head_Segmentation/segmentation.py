"""
Fetal head semantic segmentation main script
"""
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

import random
from segmentation_utils import *
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

		return input_image, mask

def normalize(input_image, mask):
	"""
	preprocessing the mask and the image
	"""
	## normalization image
	mean = tf.math.reduce_mean(input_image)
	std = tf.math.reduce_std(input_image)
	input_image = (input_image - mean)/std

	## normalization mask
	mask = mask / 255.

	return input_image, mask

@tf.function()
def random_jitter(input_image, mask, rot=False):
	"""
	Complete image preprocessing for Segmentation

	Parameters
	----------
	input_image : tensorflow tensor
		input imgage, i.e. CAM 
	mask : tensorflow tensor
		real image, i.e. US image

	Returns
	-------
	
	"""
	# Random mirroring
	if tf.random.uniform(()) > 0.5:
		input_image = tf.image.flip_left_right(input_image)
		mask = tf.image.flip_left_right(mask)

	# Random flip
	if tf.random.uniform(()) > 0.5:
		input_image = tf.image.flip_up_down(input_image)
		mask = tf.image.flip_up_down(mask)

	# Random rotation
	if rot:
		if tf.random.uniform(()) > 0.5:
			angle = np.pi/10
				
			input_image = tfa.image.rotate(input_image, tf.constant(angle))
			mask = tfa.image.rotate(mask, tf.constant(angle))
										
	return input_image, mask

def load_sample_train(sample_path):
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
	input_image, mask = random_jitter(input_image, mask, rot=False)
	input_image, mask = normalize(input_image, mask)

	return input_image, mask


def load_sample_train_rot(sample_path):
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
	input_image, mask = random_jitter(input_image, mask, rot=True)
	input_image, mask = normalize(input_image, mask)

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
	parser.add_argument("-folder_name", default='trial', type=str, help="nome of folder where save the data")
	parser.add_argument('-plot', action='store_true', help='extrapolate the feature. default=False')
	args = parser.parse_args()

	## Main folders
	train_path = 'Dataset/training_set'
	save_folder = 'Model/' + args.folder_name

	## parameters dict
	semantic_dict = semantic_dict(epochs=30)

	## MAKE tf.Dataset train and validation
	training_path = 'Dataset/training_set_stack'
	sample_list = [training_path + '/' + i for i in os.listdir(training_path)]
	random_index, sample_list = random_shuffle(sample_list)
	for i,j in zip(random_index[:10], sample_list[:10]):
		print(i,j)
	
	## splitting
	train_list, val_list, test_list = splitting_data(sample_list, splitting=(0.8,0.1,0.1))
	for aa in train_list[:10]:
		print(aa)
	print(len(train_list), len(val_list), len(test_list))

	## Cross validation
	cross_dataset = train_list + val_list  # consider all the training set for the cross validation
	kf = KFold(n_splits=3)
	kf.get_n_splits(cross_dataset)

	cross_dataset_dict = {}
	for i, (train_index, val_index) in enumerate(kf.split(cross_dataset)):
		print(f"Fold {i}:")
		cross_dataset_dict[f'train_{i}'] = [cross_dataset[j] for j in train_index]
		cross_dataset_dict[f'val_{i}'] = [cross_dataset[j] for j in val_index]
		print(train_index[:10], val_index[:10])
	print(cross_dataset_dict.keys())


	cross_vall_bool = True
	if cross_vall_bool:
		cross_folder = save_folder + '/cross'
		smart_makedir(cross_folder)
		
		for jj, loader in enumerate([load_sample_train, load_sample_train_rot]): ## cicle over the data_augumentation
			if jj == 0:
				cross_folder_save = cross_folder + '/no_rotation'
				smart_makedir(cross_folder_save)
			if jj == 1:
				cross_folder_save = cross_folder + '/rotation'
				smart_makedir(cross_folder_save)

			for i in range(3):  ## cicle over the cross-validation
			
				# delate the section
				tf.keras.backend.clear_session()

				## Create dataset
				train_dataset = tf.data.Dataset.list_files(cross_dataset_dict[f'train_{i}'], shuffle=True)
				train_dataset = train_dataset.map(loader,
															num_parallel_calls=tf.data.AUTOTUNE)
				train_dataset = train_dataset.shuffle(len(cross_dataset_dict[f'train_{i}']))
				train_dataset = train_dataset.batch(semantic_dict['batch_size'])

				val_dataset = tf.data.Dataset.list_files(cross_dataset_dict[f'val_{i}'], shuffle=True)
				val_dataset = val_dataset.map(load_sample_test,
															num_parallel_calls=tf.data.AUTOTUNE)
				val_dataset = val_dataset.batch(semantic_dict['batch_size'])
				print(i, jj, train_dataset)

				# U-net model Training
				model = unet(input_size = (224,224,1))
				
				opt = Adam(learning_rate = semantic_dict['learning_rate'], beta_1=semantic_dict['momentum'])
				model.compile(optimizer = opt, 
				            loss = "binary_crossentropy", 
							metrics = ['accuracy'])

				history = model.fit(train_dataset, verbose = 1,
							batch_size = semantic_dict['batch_size'],
							epochs = semantic_dict['epochs'],
							validation_data=val_dataset)
				
				# save only loss
				hist_accuracy = [0.] + history.history['accuracy']
				hist_val_accuracy = [0.] + history.history['val_accuracy']
				hist_loss = history.history['loss']
				hist_val_loss = history.history['val_loss']
				epochs_train = len(history.history['loss'])

				np.save(cross_folder_save + f'/random_index{i}', random_index)
				np.save(cross_folder_save + f'/history_accuracy{i}', np.array(hist_accuracy))
				np.save(cross_folder_save+ f'/history_val_accuracy{i}', np.array(hist_val_accuracy))
				np.save(cross_folder_save + f'/history_loss{i}', np.array(hist_loss))
				np.save(cross_folder_save + f'/history_val_loss{i}', np.array(hist_val_loss))
				np.save(cross_folder_save + f'/epoch_train{i}', np.array(len(history.history['loss'])))

	else:
		## Datasets
		train_dataset = tf.data.Dataset.list_files(train_list, shuffle=True)
		train_dataset = train_dataset.map(load_sample_train,
		                            	  			num_parallel_calls=tf.data.AUTOTUNE)
		train_dataset = train_dataset.shuffle(len(train_list))
		train_dataset = train_dataset.batch(semantic_dict['batch_size'])

		val_dataset = tf.data.Dataset.list_files(val_list, shuffle=True)
		val_dataset = val_dataset.map(load_sample_test,
		                            	  			num_parallel_calls=tf.data.AUTOTUNE)
		val_dataset = val_dataset.batch(semantic_dict['batch_size'])

		if args.plot : 
			dataset_visualization(train_dataset, take_ind=10)

			# data augumentation 
			for i, (img,mask) in enumerate(iter(train_dataset.take(1))):
					for i in range(1):
						rj_img, rj_mask = random_jitter(img, mask)
						fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,6), num=f'data augumentation{i}', tight_layout=True)
						ax[0].imshow(rj_img[0,:,:,:],cmap='gray')
						ax[0].axis('off')

						ax[1].imshow(rj_mask[0,:,:,:],cmap='gray')
						ax[1].axis('off')

						ax[2].imshow(rj_img[0,:,:,:],cmap='gray')
						ax[2].imshow(rj_mask[0,:,:,:],cmap='jet',alpha=0.6)
						ax[2].axis('off')

			plt.show()

		# U-net model Training
		model = unet(input_size = (224,224,1))
		print(model.summary())

		opt = Adam(learning_rate = semantic_dict['learning_rate'], beta_1=semantic_dict['momentum'])
		model.compile(optimizer = opt, 
		            loss = "binary_crossentropy", 
					metrics = ['accuracy'])

		# save only_weight
		root = save_folder + "/net_train"
		filepath = save_folder + "/weights/{epoch:02d}.hdf5"
		batch_per_epoch = int(len(train_list)/semantic_dict['batch_size'])
		print(f'batch_per_epoch: {batch_per_epoch}')

		checkPoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
									save_best_only=False, save_freq=10*batch_per_epoch)
		callbacks_list =  [checkPoint]

		history = model.fit(train_dataset, verbose = 1,
							batch_size = semantic_dict['batch_size'],
							epochs = semantic_dict['epochs'],
							validation_data=val_dataset,
							callbacks=callbacks_list)

		## save model, loss and hyperparameters
		hist_accuracy = [0.] + history.history['accuracy']
		hist_val_accuracy = [0.] + history.history['val_accuracy']
		hist_loss = history.history['loss']
		hist_val_loss = history.history['val_loss']
		epochs_train = len(history.history['loss'])

		# model.save(save_folder + '/first_try', save_format='h5')
		np.save(save_folder + '/random_index', random_index)
		np.save(save_folder + '/history_accuracy', np.array(hist_accuracy))
		np.save(save_folder + '/history_val_accuracy', np.array(hist_val_accuracy))
		np.save(save_folder + '/history_loss', np.array(hist_loss))
		np.save(save_folder + '/history_val_loss', np.array(hist_val_loss))
		np.save(save_folder + '/epoch_train', np.array(len(history.history['loss'])))

		with open(save_folder +'/summary.txt', 'w', encoding='utf-8') as file:
			model.summary(print_fn=lambda x: file.write(x + '\n'))

			for par in semantic_dict.keys():
				file.write(f'\n {par}: {semantic_dict[par]} \n ')
