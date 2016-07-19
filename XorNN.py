import os, sys
import numpy as np
from random import random
from math import exp

inputs = 2
outputs = 1
layer_size = 2
hidden_layers = 1
learning_rate = 1


#access weights
#file = open(XorNNWeights.txt,'r')

#randomly assign weights
weights = []
for l in range(1+hidden_layers):
	matrix_rows = layer_size
	if l==hidden_layers: matrix_rows = outputs
	matrix_cols = layer_size
	if l==0: matrix_cols = inputs
	matrix = ()
	for r in range(matrix_rows):
		row = ()
		for c in range(matrix_cols):
			row += (random(),)
		matrix += (row,)
	weights.append(np.matrix(matrix))

def print_weights():
	for l in range(1+hidden_layers):
		print("Layer: ",l)
		print(weights[l])
	
def sigmoid(z):
	return 1/(1+exp(-z))
def serial_derivative(z):
	return sigmoid(z)*(1-sigmoid(z))
	
transfer = np.vectorize(sigmoid)
derivative = np.vectorize(serial_derivative)
	
def output(arr):
	vector = np.matrix(arr).T
	for l in range(1+hidden_layers):
		vector = weights[l] * vector
		vector = transfer(vector)
	return vector
	
def back_propagate(arr_in, arr_target):
	global weights
	weighted_input = []
	activation = []
	gradient = []
	
	#feed forward
	vector = np.matrix(arr_in).T
	#activation.append(vector)
	for l in range(1+hidden_layers):
		vector = weights[l] * vector
		weighted_input.append(vector)
		vector = transfer(vector)
		activation.append(vector)
		gradient.append(np.matrix(np.zeros(len(weights[l]))).T)
		
	#back propagate
	difference = activation[hidden_layers] - np.matrix(arr_target)
	gradient[hidden_layers] = np.multiply(difference, derivative(weighted_input[hidden_layers]))
	for l in range(hidden_layers-1,-1,-1):
		gradient[l] = np.multiply(weights[l+1].T*gradient[l+1], derivative(weighted_input[l]))
	
	#update weights
	for l in range(1+hidden_layers):
		'''print("gradient",l)
		print(gradient[l])'''
		diff = gradient[l]*(activation[l].T)
		diff = (learning_rate/len(weights[l])) * diff
		weights[l] = weights[l] - diff
	
print_weights()
print("(0,0)",output((0,0)))
print("(1,0)",output((1,0)))
print("(0,1)",output((0,1)))
print("(1,1)",output((1,1)))
for i in range(10000):
    back_propagate((0,0),(0))
    back_propagate((1,0),(1))
    back_propagate((0,1),(1))
    back_propagate((1,1),(0))
print_weights()
print("(0,0)",output((0,0)))
print("(1,0)",output((1,0)))
print("(0,1)",output((0,1)))
print("(1,1)",output((1,1)))

		