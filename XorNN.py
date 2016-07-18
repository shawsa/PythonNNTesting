import os, sys
import numpy as np
from random import random
from math import exp

inputs = 2
outputs = 1
layer_size = 2
hidden_layers = 1


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
	
transfer = np.vectorize(sigmoid)
	
def output(arr):
	vector = np.matrix(arr).T
	for l in range(1+hidden_layers):
		print(vector)
		vector = weights[l] * vector
		vector = transfer(vector)
	return vector