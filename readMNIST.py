import struct
import pickle
from neuralnetwork import *

file = open("MNIST-training-labels.idx",'rb')

labels = []

magic_word = file.read(4)
total_label_records = struct.unpack('>i',file.read(4))[0]

print('extracting labels...')
for i in range(total_label_records):
	labels.append(struct.unpack('B',file.read(1))[0])
	

file = open("MNIST-training-images.idx",'rb')

images = []
magic_word = file.read(4)
total_image_records = struct.unpack('>i',file.read(4))[0]
row_size = struct.unpack('>i',file.read(4))[0]
col_size = struct.unpack('>i',file.read(4))[0]


#testing
#total_image_records = 10

print('extracting images...')
for i in range(total_image_records):
	image = []
	for j in range(row_size*col_size):
		image.append(struct.unpack('B',file.read(1))[0])
	images.append(image)
	
def print_image(i):
	for r in range(row_size):
		for c in range(col_size):
			str = " "
			if images[i][r*col_size+c] > 150: str = 'x'
			print(str,end='')
		print('')
		
#format labels for Neural Network
expected = []
for i in range(total_label_records):
	l = [0] * 10
	l[labels[i]] = 1
	expected.append(l)

nn = NeuralNetwork(28*28,10,(500))

def interpret_nn(arr_output):
	max = -1
	index = 0
	for i in range(10):
		if arr_output[i] > max: 
			index = i
			max = arr_output[i]
	return index


def run_training_set(learning_rate):
	print("Training")
	for i in range(total_image_records):
		print(i,end='\r')
		nn.back_prop(images[i],expected[i],learning_rate)
	print(total_image_records)
	hits = 0
	print("Calculating accuracy...",end="\r")
	for i in range(total_image_records):
		if labels[i] != interpret_nn(nn.output(images[i])):
			hits += 1
	print(hits/total_image_records*100, "%                      ")
	return hits/total_image_records
		
def run_and_save(required_accuracy):
	accuracy = .5
	run = 0
	while accuracy < required_accuracy:
		learning_rate = 2*(1-accuracy)
		accuracy = run_training_set(learning_rate)
		pickle.dump(nn, open("DigitRecog " + str(run) + ' ' + str('{0:.2f'.format(accuracy)),"wb"))
		print("Accuracy: ",accuracy)
		run += 1