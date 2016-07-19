import numpy as np
from random import random

def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))
def serial_derivative(z):
	return sigmoid(z)*(1-sigmoid(z))
	
transfer = np.vectorize(sigmoid)
derivative = np.vectorize(serial_derivative)

class NeuralNetwork:
	weights = []
	bias = []
	def __init__(self,inputs, outputs, hidden_layers, hidden_layer_size):
		self.inputs = inputs
		self.hidden_layers = hidden_layers
		self.hidden_layer_size = hidden_layer_size
		self.outputs = outputs
		weights = []
		bias = []
		for l in range(1+hidden_layers):
			matrix_rows = hidden_layer_size
			if l==hidden_layers: matrix_rows = outputs
			matrix_cols = hidden_layer_size
			if l==0: matrix_cols = inputs
			matrix = ()
			bias_vector = ()
			for r in range(matrix_rows):
				row = ()
				for c in range(matrix_cols):
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
	
	def print_network(self):
		for l in range(self.hidden_layers+1):
			print("Layer ",l)
			print("weights")
			print(self.weights[l])
			print("bias")
			print(self.bias[l])
	
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
			#placeholder for gradient
			gradient.append(np.matrix(np.zeros(vector.size)).T)
		#we would like activation[-1] to contain the input. In python appeding
		#the input to the end of the list acomplishes this.
		activation.append(np.matrix(arr_inputs).T)
		
		#back propagate
		gradient[self.hidden_layers] = np.multiply(vector - np.matrix(arr_outputs).T, 
			derivative(weighted_inputs[self.hidden_layers]))
		'''print("target: ",arr_outputs)
		print("output: ",vector)
		print("gradient")
		print(gradient[self.hidden_layers])'''
		
		for l in range(self.hidden_layers-1,-1,-1):
			gradient[l] = np.multiply(self.weights[l+1].T * gradient[l+1], derivative(weighted_inputs[l]))
		
		#update weights
		for l in range(self.hidden_layers + 1):
			'''print("Layer ",l)
			print("activation")
			print(activation[l])
			print("weighted inputs")
			print(weighted_inputs[l])
			print("gradient")
			print(gradient[l])'''
			self.weights[l] = self.weights[l] - learning_rate * (gradient[l]*(activation[l-1].T))
			self.bias[l] = self.bias[l] - learning_rate * gradient[l]
		#print("new: ", self.output(arr_inputs))
			