import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return sigmoid(x) * (1 - sigmoid(x))
	
class FeedForwardNeuralNetwork:
	def __init__(self, input_size, hidden_size, output_size):
		
		# Initialise weights and biases
		self.W1 = np.random.randn(input_size, hidden_size)
		self.b1 = np.zeros((1, hidden_size))
		
		self.W2 = np.random.randn(hidden_size, output_size)
		self.b2 = np.zeros((1, output_size))

	# The mean squared error function
	def lossFunction(self, y_true, y_pred):
		return np.mean(np.square(y_true - y_pred))
	
	# returns the loss using mean squared 
	def calculate_loss(self, X, Y):
		output = self.forward_pass(X)
		loss = self.lossFunction(Y, output)
		return loss
	
	def forward_pass(self, X):
		# Forward pass, just go through the layers multiplying the weights + adding bias
		# Return the final output self.a2
		
		self.z1 = X.dot(self.W1) + self.b1
		self.a1 = sigmoid(self.z1)
		
		self.z2 = self.a1.dot(self.W2) + self.b2
		self.a2 = sigmoid(self.z2)
		return self.a2
	
	def backward_pass(self, X, Y, learning_rate):
		m = X.shape[0]
		
		# Calculate gradient of the loss function with respect to a2
		# loss function is in this case Mean squared error, this is the derivative of it
		loss_derivative = 2 * (self.a2 - Y) / m
		
		# Calculate gradients for W2 and b2
		dW2 = self.a1.T.dot(loss_derivative)
		db2 = np.sum(loss_derivative, axis=0)
		
		# Calculate gradient for the hidden layer
		d_hidden = loss_derivative.dot(self.W2.T) * sigmoid_derivative(self.z1)
		
		# Calculate gradients for W1 and b1
		dW1 = X.T.dot(d_hidden)
		db1 = np.sum(d_hidden, axis=0)
		
		# Update the weights and biases
		self.W1 -= learning_rate * dW1
		self.b1 -= learning_rate * db1
		self.W2 -= learning_rate * dW2
		self.b2 -= learning_rate * db2
		
		
	def train(self, X_train, Y_train, X_valid, Y_valid, epochs, mini_batch_size, learning_rate):
		for epoch in range(epochs):
			
			# going through all batches
			for i in range(0, len(X_train), mini_batch_size):
				mini_batch_X = X_train[i:i+mini_batch_size, :]
				mini_batch_Y = Y_train[i:i+mini_batch_size]
				
				self.forward_pass(mini_batch_X)
				self.backward_pass(mini_batch_X, mini_batch_Y, learning_rate)
				
			validation_loss = self.calculate_loss(X_valid, Y_valid)
			print(f"Epoch {epoch+1}/{epochs} completed, validation loss: {validation_loss}")
			
	def test(self, test_X, test_y):
		# Perform a forward pass on the test data
		test_output = self.forward_pass(test_X)
		
		# Get the index of the maximum value to find the predicted digit
		# values are probabilities
		test_predictions = np.argmax(test_output, axis=1)
		
		# Calculate the accuracy of the network
		accuracy = np.mean(test_predictions == test_y) * 100
		return accuracy
