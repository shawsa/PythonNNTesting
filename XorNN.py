from neuralnetwork import *
import pickle
fileObject = open('Xor.nn','rb')
xornn = pickle.load(fileObject)

def xor(a,b):
	value = xornn.output((a,b))
	ret = False
	if value > .5: ret = True
	return ret
