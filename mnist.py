import struct
import pickle
from neuralnetwork import *

def read_training_labels(*max_records):
	file = open("MNIST-training-labels.idx",'rb')
	labels = []
	magic_word = file.read(4)
	total_label_records = struct.unpack('>i',file.read(4))[0]
	if max_records:
		total_label_records = min(total_label_records,max_records[0])
	for i in range(total_label_records):
		labels.append(struct.unpack('B',file.read(1))[0])
	return labels
	
def read_testing_labels(*max_records):
	file = open("MNIST-testing-labels.idx",'rb')
	labels = []
	magic_word = file.read(4)
	total_label_records = struct.unpack('>i',file.read(4))[0]
	if max_records:
		total_label_records = min(total_label_records,max_records[0])
	for i in range(total_label_records):
		labels.append(struct.unpack('B',file.read(1))[0])
	return labels
	
def read_training_images(*max_records):
	file = open("MNIST-training-images.idx",'rb')
	images = []
	magic_word = file.read(4)
	total_image_records = struct.unpack('>i',file.read(4))[0]
	row_size = struct.unpack('>i',file.read(4))[0]
	col_size = struct.unpack('>i',file.read(4))[0]
	if max_records:
		total_image_records = min(total_label_records,max_records[0])
	for i in range(total_image_records):
		image = []
		for j in range(row_size*col_size):
			image.append(struct.unpack('B',file.read(1))[0])
		images.append(image)
	return images
	
def read_testing_images(*max_records):
	file = open("MNIST-testing-images.idx",'rb')
	images = []
	magic_word = file.read(4)
	total_image_records = struct.unpack('>i',file.read(4))[0]
	row_size = struct.unpack('>i',file.read(4))[0]
	col_size = struct.unpack('>i',file.read(4))[0]
	if max_records:
		total_image_records = min(total_label_records,max_records[0])
	for i in range(total_image_records):
		image = []
		for j in range(row_size*col_size):
			image.append(struct.unpack('B',file.read(1))[0])
		images.append(image)
	return images
	
def format_labels(labels):
	expected = []
	for i in range(len(labels)):
		l = [0] * 10
		l[labels[i]] = 1
		expected.append(l)
	return expected
	
def print_image(img):
	for r in range(28):
		for c in range(28):
			str = " "
			if img[r*28+c] > 150: str = 'x'
			print(str,end='')
		print('')
		
def interpret_nn(arr_output):
	max = -1
	index = 0
	for i in range(10):
		if arr_output[i] > max: 
			index = i
			max = arr_output[i]
	return index
	
def run_training_set(nn, images, labels, learning_rate):
	expected = format_labels(labels)
	assert len(images) == len(expected)
	for i in range(len(images)):
		print(i,end='\r')
		nn.back_prop(images[i],expected[i],learning_rate)
		
def accuracy(nn, images, labels):
	hits = 0
	for i in range(len(images)):
		if labels[i] == interpret_nn(nn.output(images[i])):
			hits += 1
	return hits/len(images)
	
def run_and_save(trial, nn, images, labels, required_accuracy, images_testing, labels_testing):
	print("Pre-training summary")
	print("Training set accuracy: ", '0:.2f'.format(accuracy(nn, images, labels)*100) + '%')
	print("Testing set accuracy: ", '0:.2f'.format(accuracy(nn, images_testing, labels_testing)*100) + '%')
	
	acc = .5
	run = 0
	learning_rate = 3
	while acc < required_accuracy:
		print("Run",run)
		#learning_rate = 2*(1-accuracy)
		run_training_set(nn, images, labels, learning_rate)
		pickle.dump(nn, open("DigitRecog Trial " + str(trial) + ' run ' + str(run) + ' accuracy ' + str('{0:.2f}'.format(accuracy)) + '.nn',"wb"))
		acc = accuracy(nn, images, labels)
		print("Training set accuracy: ", '{0:.2f}'.format(acc*100) + '%')
		print("Testing set accuracy: ", '{0:.2f}'.format(accuracy(nn, images_testing, labels_testing)*100) + '%')
		run += 1