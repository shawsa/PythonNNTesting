import struct
from neuralnetwork import *

file = open("MNIST-training-lables.idx",'rb')

lables = []

magic_word = file.read(4)
total_lable_records = struct.unpack('>i',file.read(4))[0]

print('extracting lables...')
for i in range(total_lable_records):
	lables.append(struct.unpack('B',file.read(1))[0])
	

file = open("MNIST-training-images.idx",'rb')

images = []
magic_word = file.read(4)
total_image_records = struct.unpack('>i',file.read(4))[0]
row_size = struct.unpack('>i',file.read(4))[0]
col_size = struct.unpack('>i',file.read(4))[0]


#testing
total_image_records = 10

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
		
#format lables for Neural Network
expected = []
for i in range(total_lable_records):
	l = [0] * 10
	l[lables[i]] = 1
	expected.append(l)

nn = NeuralNetwork(28*28,10,5,300)

def interpret_nn(arr_output):
	max = -1
	index = 0
	for i in range(10):
		if arr_output[i] > max: 
			index = i
			max = arr_output[i]
	return index

'''print("Training")
for i in range(total_image_records):
	print(i,end='\r')
	nn.back_prop(images[i],expected[i],.05)'''