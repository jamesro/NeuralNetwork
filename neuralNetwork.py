import numpy as np
import matplotlib.pyplot as plt
from cPickle import dump,load

def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoidGradient(z):
	g = sigmoid(z)
	return g*(1-g)

def Einit(Lin,Lout):
	return np.sqrt(6)/(np.sqrt(Lin + Lout))


class neuralNetwork:
	def __init__(self, inputData, targets, hiddenLayers):

		self.nIn = np.shape(inputData)[1]
		self.nOut = np.shape(targets)[1]
		self.nHid = hiddenLayers

		self.m = np.shape(inputData)[0]

		# Initialise thetas to break symmetry
		E = Einit(self.nIn,self.nHid)
		self.theta1 = np.random.uniform(-E,E,(self.nHid,self.nIn+1))
		E = Einit(self.nHid,self.nOut)
		self.theta2 = np.random.uniform(-E,E,(self.nOut,self.nHid+1))

	def fProp(self,X):
		# Calculate hidden activations
		self.zHidden = np.dot(X,np.transpose(self.theta1))
		self.aHidden = sigmoid(self.zHidden)

		# Add 1 to each (hidden)layer
		self.aHidden = np.concatenate((np.ones((self.m,1)),self.aHidden),axis=1)

		#Return output activations
		self.zOut = np.dot(self.aHidden,np.transpose(self.theta2))
		return sigmoid(self.zOut)
	
	def mini_batch(self,inputData,targets,batch_size,iterations,eta,momentum=0.9):
		for n in range(iterations):
			self.m = np.shape(inputData)[0]
			for i in range(0,self.m, batch_size):
				batch_Data = inputData[i:i+batch_size,:]
				batch_targets = targets[i:i+batch_size,:]

				# (Temprorarily) change self.m to the size of the new batch of data
				self.m = np.shape(batch_Data)[0]

				self.train(batch_Data,batch_targets,iterations=1,eta=eta,momentum=momentum)

			# Suffle input data and target data in unison for each full iteration
			shuffle_length = np.shape(inputData)[0]
			shuffle_index = range(shuffle_length)
			np.random.shuffle(shuffle_index)
			for i in range(shuffle_length):
				inputData[i] = inputData[shuffle_index[i]]
				targets[i] = targets[shuffle_index[i]]
			

	def train(self,inputData,targets,iterations,eta,momentum=0.9,showPlots=False):
		self.momentum = momentum

		# Add a column of ones to the input data matrix 
		inputData = np.concatenate((np.ones((self.m,1)),inputData),axis = 1)
		
		theta1_grad = np.zeros(np.shape(self.theta1))
		theta2_grad = np.zeros(np.shape(self.theta2))
		if showPlots:
			J = np.zeros(iterations)
		for n in range(iterations):			
			self.aOut = self.fProp(inputData)
			
			J1 = -np.multiply(targets,np.log(self.aOut))
			J2 = -np.multiply(1-targets,np.log(1-self.aOut))
			Jreg = 0
			cost = (1.0/self.m)*np.sum(J1 + J2) + Jreg
			if showPlots:
				J[n] = cost

			print(" ".join(("Iteration:",str(n), "	Error:",str(cost))))

			self.deltaOut = (self.aOut - targets)
			self.deltaHidden = np.multiply(np.dot(self.deltaOut,self.theta2[:,1:]),sigmoidGradient(self.zHidden))

			self.DeltaHidden = np.dot(np.transpose(self.deltaHidden),inputData)
			self.DeltaOut = np.dot(np.transpose(self.deltaOut),self.aHidden)

			Theta1_grad = (1.0/self.m)*self.DeltaHidden + self.momentum*theta1_grad
			Theta2_grad = (1.0/self.m)*self.DeltaOut + self.momentum*theta2_grad
			
			self.theta1 -= eta*Theta1_grad
			self.theta2 -= eta*Theta2_grad

		if showPlots:
			plt.plot(J)
			plt.ylabel("Cost")
			plt.xlabel("Iterations")
			plt.show()
		else:
			print("Training complete")


	def predict(self,inputData):
		self.m = np.shape(inputData)[0]
		inputData = np.concatenate((np.ones((self.m,1)),inputData),axis = 1)
		return np.round(self.fProp(inputData))

	def save(self,filename):
		"""Uses cPickle to save the values of the weights (theta) in the network).
			cPickle is up to 1000 times faster than pickle. Use .pkl as the file extension"""
		Theta = [self.theta1,self.theta2]
		dump(Theta,open(filename,"wb"))

	def load(self,filename):
		"""Loads the saved values of theta (the weights of the network)"""
		Theta = load(open(filename,"rb"))
		self.theta1,self.theta2 = Theta

def demo():
	# Logical XOR operator
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	y = np.array([[0],[1],[1],[0]])
		
	n = neuralNetwork(X,y,2)	
	n.train(X,y,10000,0.25,momentum=0.95,showPlots=False)	







if __name__ == "__main__":
	demo()