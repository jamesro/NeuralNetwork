from MNIST import *
from nn3 import *

# ------------------------------------------------------------------------------------
# Training Set
images, labels = load_mnist('training',path=os.getcwd())
N, nrows, ncols = images.shape
images = images.reshape(N,-1,nrows,ncols).swapaxes(1,2).reshape(N,ncols*nrows)

labels = labels.reshape(N,1)
labelMatrix = np.zeros((N,10))
for i in range(N):
	labelMatrix[i] = np.concatenate((np.zeros(labels[i][0]),[1],np.zeros(9-labels[i][0])),1)

n = neuralNetwork(images,labelMatrix,nrows*ncols)


n.mini_batch(images,labelMatrix,batch_size=100,iterations=10,eta=0.25,momentum=0.9)

# Percentage of correct classifications for training set
# Ptrain = np.sum(np.sum(n.predict(images)==labelMatrix,axis=1)/10.0)/(N*1.0)



# ------------------------------------------------------------------------------------
# Test Set
images, labels = load_mnist('testing',path=os.getcwd())
N, nrows, ncols = images.shape
images = images.reshape(N,-1,nrows,ncols).swapaxes(1,2).reshape(N,ncols*nrows)

labels = labels.reshape(N,1)
labelMatrix = np.zeros((N,10))
for i in range(N):
	labelMatrix[i] = np.concatenate((np.zeros(labels[i][0]),[1],np.zeros(9-labels[i][0])),1)

# Percentage of correct classifications for testing set
Ptest = np.sum(np.sum(n.predict(images)==labelMatrix,axis=1)/10.0)/(N*1.0)

print Ptest