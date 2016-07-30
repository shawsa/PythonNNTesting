import numpy as np
from random import random

def sigmoid(z):
		if z > 100: return 1
		if z < -100: return 0
		return 1.0/(1.0+np.exp(-z))
def serial_derivative(z):
	return sigmoid(z)*(1-sigmoid(z))
	
transfer = np.vectorize(sigmoid)
derivative = np.vectorize(serial_derivative)

class NeuralNetwork:
	weights = []
	bias = []
	def __init__(self,inputs, outputs, hidden_layers):
		self.inputs = inputs
		self.hidden_layers = len(hidden_layers)
		self.outputs = outputs
		weights = []
		bias = []
		#weights between inputs and first hidden layer
		matrix = ()
		bias_vector = ()
		for r in range(hidden_layers[0]):
			row = ()
			for c in range(inputs):
				row += (2*random()-1,)
			matrix += (row,)
			bias_vector += (2*random()-1,)
		weights.append(np.matrix(matrix))
		bias.append(np.matrix(bias_vector).T)
		#weights between hidden layers
		for l in range(len(hidden_layers)-1):
			matrix = ()
			bias_vector = ()
			for r in range(hidden_layers[l+1]):
				row = ()
				for c in range(hidden_layers[l]):
					row += (2*random()-1,)
				matrix += (row,)
				bias_vector += (2*random()-1,)
			weights.append(np.matrix(matrix))
			bias.append(np.matrix(bias_vector).T)
		#weights between last hidden layer and output
		matrix = ()
		bias_vector = ()
		for r in range(outputs):
			row = ()
			for c in range(hidden_layers[-1]):
				row += (2*random()-1,)
			matrix += (row,)
			bias_vector += (2*random()-1,)
		weights.append(np.matrix(matrix))
		bias.append(np.matrix(bias_vector).T)
		self.weights = weights
		self.bias = bias
		
	def output(self, arr_inputs):
		vector = np.matrix(arr_inputs).T
		for l in range(self.hidden_layers+1):
			vector = transfer(self.weights[l] * vector + self.bias[l])
		return vector
	
	def print_weights(self):
		for l in range(self.hidden_layers+1):
			print("Layer ",l)
			print("weights")
			print(self.weights[l])
			print("bias")
			print(self.bias[l])
			
	def get_structure(self):
		structure = []
		structure.append(self.weights[0].shape[1])
		for l in range(self.hidden_layers):
			structure.append(self.weights[l].shape[0])
		structure.append(self.weights[self.hidden_layers].shape[0])
		return structure
	
	def back_prop(self, arr_inputs, arr_outputs, learning_rate):
		activation = []
		weighted_inputs = []
		gradient = []
		
		#feed forward
		vector = np.matrix(arr_inputs).T
		for l in range(self.hidden_layers + 1):
			vector = self.weights[l] * vector + self.bias[l]
			weighted_inputs.append(vector)
			vector = transfer(vector)
			activation.append(vector)
			gradient.append(np.matrix(np.zeros(vector.size)).T)
		#we would like activation[-1] to contain the input. In python appeding
		#the input to the end of the list acomplishes this.
		activation.append(np.matrix(arr_inputs).T)
		
		#back propagate
		gradient[self.hidden_layers] = np.multiply(vector - np.matrix(arr_outputs).T, 
			derivative(weighted_inputs[self.hidden_layers]))
		
		for l in range(self.hidden_layers-1,-1,-1):
			gradient[l] = np.multiply(self.weights[l+1].T * gradient[l+1], derivative(weighted_inputs[l]))
		
		#update weights
		for l in range(self.hidden_layers + 1):
			self.weights[l] = self.weights[l] - learning_rate * (gradient[l]*(activation[l-1].T))
			self.bias[l] = self.bias[l] - learning_rate * gradient[l]
			