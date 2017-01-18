import numpy as np
import pickle
def nonlin(x,deriv=False):    #creating the sigmoid function
	if(deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))

#input data
X=np.array([[0,0,1],
	   [0,1,1],	
	   [1,0,1],
	   [1,1,1]])

#output data
y=np.array([[0],
	    [1],
	    [0],
	    [1]])

np.random.seed(1)


#synapses
syn0 = 2*np.random.random((3,4))-1
syn1 = 2*np.random.random((4,1))-1

 
#training step
def train_model(X,y,syn0,syn1):
	for j in xrange(100000):
	
		l0=X #layer 0:input layer
		l1=nonlin(np.dot(l0, syn0)) #layer 1:hidden layer
		l2=nonlin(np.dot(l1, syn1)) #layer 2:output layer
	
		l2_error = y-l2 
		
		if(j % 1000)==0:   
			print "Error:" + str(np.mean(np.abs(l2_error)))

		l2_delta = l2_error*nonlin(l2, deriv=True)
		
		l1_error = l2_delta.dot(syn1.T)
		
		l1_delta = l1_error*nonlin(l1, deriv=True) 
		
	#update weights
		syn1 +=l1.T.dot(l2_delta)
		syn0 +=l0.T.dot(l1_delta)
	return l2

i=train_model(X,y,syn0,syn1)

with open('ann.pickle','wb') as temp:
	pickle.dump(train_model, temp)
pickle_in = open('ann.pickle','rb')
train_model=pickle.load(pickle_in)

print "Output after training"
print i 

