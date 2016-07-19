from BackPropNN import *
import pickle
nn = BackPropNN(2,1,1,3,.05)
#nn.print_network()
print("Outputs")
print("(0,0)", nn.output((0,0)))
print("(1,0)", nn.output((1,0)))
print("(0,1)", nn.output((0,1)))
print("(1,1)", nn.output((1,1)))
print("Training...")
learning_rate = .5
for i in range(20000):
	if i%2000==0: 
		learning_rate /= 2
		print("learning rate: ", learning_rate)
	nn.back_prop((0,0),(0))
	nn.back_prop((1,0),(1))
	nn.back_prop((0,1),(1))
	nn.back_prop((1,1),(0))
#nn.print_network()
print("Outputs")
print("(0,0)", nn.output((0,0)))
print("(1,0)", nn.output((1,0)))
print("(0,1)", nn.output((0,1)))
print("(1,1)", nn.output((1,1)))
#fileObject = open('Xor.nn','wb')
#pickle.dump(nn,fileObject)