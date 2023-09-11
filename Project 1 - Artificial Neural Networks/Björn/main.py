from data_loader import load_data
from FFNeuralNetwork import FeedForwardNeuralNetwork

def main():
	# Hyperparameters
	input_size = 784
	hidden_size = 30
	output_size = 10
	epochs = 15
	mini_batch_size = 30
	learning_rate = 0.1
	
	# Initialise network
	nn = FeedForwardNeuralNetwork(input_size, hidden_size, output_size)
	
	# Load data
	train_X, train_y = load_data('mnist.pkl', 'train')
	test_X, test_y = load_data('mnist.pkl', 'test')
	val_X, val_y = load_data('mnist.pkl', 'validation')
	
	print("Training the network...")
	nn.train(train_X, train_y, val_X, val_y, epochs, mini_batch_size, learning_rate)
	
	print("Testing the network...")
	accuracy = nn.test(test_X, test_y)
	
	print("Total accuracy after training: ", accuracy)
	
if __name__ == "__main__":
	main()
