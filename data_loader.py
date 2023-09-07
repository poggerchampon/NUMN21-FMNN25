import pickle
import numpy as np

def load_training_data(file_name):
	with open(file_name, 'rb') as f:
		file_content = pickle.load(f, encoding='latin1')
	
	# Unpack data
	train_data, validation_data, test_data = file_content
	train_X, train_y = train_data
	
	# Convert labels to one hot vectors
	train_y = convert_to_one_hot(train_y, 10)
	
	return train_X, train_y

def load_test_data(file_name):
	with open(file_name, 'rb') as f:
		file_content = pickle.load(f, encoding='latin1')
		
	# Unpack data
	train_data, validation_data, test_data = file_content
	test_X, test_y = test_data
	
	return test_X, test_y

def load_validation_data(file_name):
	with open(file_name, 'rb') as f:
		file_content = pickle.load(f, encoding='latin1')
		
	# Unpack data
	train_data, validation_data, test_data = file_content
	val_X, val_y = validation_data
	
	# Convert labels to one hot vectors
	val_y = convert_to_one_hot(val_y, 10)
	
	return val_X, val_y

# convert labels to one hot vectors, e.g 2 becomes [0 0 1 0 0 0 0 0 0 0]
def convert_to_one_hot(y, num_classes):
	one_hot = np.zeros((y.shape[0], num_classes))
	one_hot[np.arange(y.shape[0]), y] = 1
	return one_hot