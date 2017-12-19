import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline


mean_01 = np.asarray([0., 2.])
sigma_01 = np.asarray([[1.0, 0.0], [0.0, 1.0]])

mean_02 = np.asarray([4., 0.])
sigma_02 = np.asarray([[1.0, 0.0], [0.0, 1.0]])

print mean_01
print sigma_01

data_01 = np.random.multivariate_normal(mean_01, sigma_01, 500)
data_02 = np.random.multivariate_normal(mean_02, sigma_02, 500)
print data_01.shape,data_02.shape

plt.figure(0) 
plt.xlim(-4,10)
plt.ylim(-4,6)
plt.grid('on')
plt.scatter(data_01[:, 0], data_01[:, 1], color='red')
plt.scatter(data_02[:, 0], data_02[:, 1], color='green')
plt.show()



labels = np.zeros((1000,1))
labels[500:,:] = 1.0

data = np.concatenate([data_01,data_02],axis = 0)
print data.shape

ind = range(1000)
np.random.shuffle(ind)

print ind[:10]
#shuffling
data = data[ind]
labels = labels[ind]
print data.shape,labels.shape

def distance(x1,x2):
	return np.sqrt(((x1-x2)**2).sum())

def knn(x, train, targets, k = 5):
	m = train.shape[0]
	dist = []
	for ix in range(m):
		pass
		dist.append(distance(x,train[ix]))
	dist = np.asarray(dist) #convert to numpy array
	indx = np.argsort(dist)
	# dist[indx] will be sorted
	#  print labels[indx] # max of this is answer
	sorted_labels = labels[indx][:k] 
	# print sorted_labels

	#list of unique values and their count
	counts = np.unique(sorted_labels,return_counts = True)
	return counts[0][np.argmax(counts[1])]
	# unique nos : count of nos


x_test = np.asarray([2.0, 0.0])
knn(x_test,data,labels) 


#accuracy
split = int(data.shape[0]*0.75)

X_train = data[:split]
X_test = data[split:]

Y_train = labels[:split]
Y_test = labels[split:]

print X_train.shape,X_test.shape
print Y_train.shape,Y_test.shape


preds = []
#calculating for 250 testing vectors
for tx in range(X_test.shape[0]):
    preds.append(knn(X_test[tx], X_train, Y_train))
preds = np.asarray(preds).reshape((250, 1))
print preds.shape


print 100*(preds == Y_test).sum()/float(preds.shape[0])