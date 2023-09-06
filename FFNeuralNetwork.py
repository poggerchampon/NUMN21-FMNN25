import numpy as np
import pickle

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return sigmoid(x) * (1 - sigmoid(x))

def lossFunction(self, y_true, y_pred):
	return np.mean(np.square(y_true - y_pred))

def readData(file):
	with open('mnist.pkl', 'rb') as f:
		mnist_data = pickle.load(f, encoding='latin1')
		
	return mnist_data
	
class FeedForwardNeuralNetwork:
	def __init__(self, input_size, hidden_size, output_size):		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		
		#initiate weights...
				
	def forward_pass(self, x):
		# Forward pass...
		return 0
	
	def backward_pass(self, X, Y, learning_rate):
		# Backpropagation...
		return 0
		
	def train(self, X_train, Y_train, X_valid, Y_valid, epochs, mini_batch_size, learning_rate):
		for epoch in range(epochs):
				
			for i in range(0, len(X_train), mini_batch_size):
				mini_batch_X = X_train[:, i:i+mini_batch_size]
				mini_batch_Y = Y_train[i:i+mini_batch_size]
				
				self.forward_pass(mini_batch_X)
				self.backward_pass(mini_batch_X, mini_batch_Y, learning_rate)
	

input_size = 784
hidden_size = 30
output_size = 10

train_data, validation_data, test_data = readData('mnist.pkl')

# Unpack training data
train_X, train_y = train_data
train_X = train_X.T

# Unpack validation data
validation_X, validation_y = validation_data
validation_X = validation_X.T

# Unpack test data
test_X, test_y = test_data
test_X = test_X.T

nn = FeedForwardNeuralNetwork(input_size, hidden_size, output_size)