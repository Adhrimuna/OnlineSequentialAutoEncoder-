import numpy as np

def load_mnist(path):
	mnist_dataset = np.load(path)
	return mnist_dataset['x_train'], mnist_dataset['x_test'], mnist_dataset['y_train'], mnist_dataset['y_test']
