import pickle
import numpy as np

# Returns the images and labels for the type: 'data:type'
def load_data(file_name: str, data_type: str):
	# Attempt data read from specified data file
	try:
		with open(file_name, 'rb') as f:
			file_content = pickle.load(f, encoding='latin1')
	except (FileNotFoundError, pickle.PickleError):
		print(f"An error occured when opening the file: {file_name}")
		return None, None
	
	# Handle the 3 data types
	# Convert Train & Val to one hot vectors (for calculating loss gradient)
	if data_type == 'train':
		X, y = file_content[0]
		y = convert_to_one_hot(y, 10)
	elif data_type == 'validation':
		X, y = file_content[1]
		y = convert_to_one_hot(y, 10)
	elif data_type == 'test':
		X, y = file_content[2]
	else:
		print(f"Invalid data type: {data_type}")
		return None, None
	
	return X, y

# Convert labels to one hot vectors, e.g 2 becomes [0 0 1 0 0 0 0 0 0 0]
def convert_to_one_hot(y, num_classes):
	one_hot = np.zeros((y.shape[0], num_classes))
	one_hot[np.arange(y.shape[0]), y] = 1
	return one_hot
