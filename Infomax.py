import tensorflow as tf
import soundfile as sf
import numpy as np
import time
import cProfile

#read data, the type of data is a 1-D np.ndarray
# data1, fs1 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig1.wav')
# data2, fs2 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig2.wav')

#Windows reading path
data1, fs1 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig1.wav')
data2, fs2 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig2.wav')

#this sets the random seed to a fixed number.
np.random.seed(10)

n_sources = 2
batch_size = 500

#randomly initialize the mixing matrix A
#each entry is from uniform[0,1), 
A = np.random.rand(2,2)

#the number of data points. Also the number of columns.
#Ns = len(data1)
Ns = 50000 #self defined data length
data1 = data1[:Ns]
data2 = data2[:Ns]

#stack the two data arrays together as the source signals
#the shape of S is (2,Ns)
S = np.array((data1,data2))

#V is the observed signal mixture.
V = np.dot(A,S)

#Remove mean
#To take the mean of each row, choose axis = 1
meanValue = np.mean(V, axis = 1)
#This changes meanValue from 1d to 2d, now a column vector with size dimension*1
meanValue = np.reshape(meanValue,(len(meanValue),1))
#This creates an array full of ones with the same length as the column number of V
oneArray = np.ones((1,Ns))
#This creates a matrix full of mean values for each row
meanMatrix = np.dot(meanValue,oneArray)
#This gives V zero mean
V = V - meanMatrix

#whitening
#this computes the covariance matrix of V. Each row should be a variable and each column should be an observation.
covMatrix = np.cov(V)
#this gets the svd form of the covMatrix.
P,d,Qt = np.linalg.svd(covMatrix, full_matrices=False)
Q = Qt.T
#this gets the first L entries
d = d[:n_sources]
D = np.diag(d)
#this gets the first L columns of singular (eigen) vectors
E = P[:,:n_sources]
#this computes the whitening matrix D^(-1/2)*E.T
whiteningMatrix = np.dot(np.linalg.inv(np.sqrt(D)),E.T)
#whitened is the whitened signal matrix
whitened = np.dot(whiteningMatrix,V)

data = whitened

#None means it can be any value
x = tf.placeholder('float', [n_sources, None])

#This give s a random block of data with size num
def next_batch(num, data):

    #Return a total of `num` random samples and labels. 

    idx = np.arange(0 , len(data[0])-num)
    np.random.shuffle(idx)
    idx = idx[0]
    #This gives num random columns of the data array
    data_shuffle = data[:,idx:idx+num]

    return np.asarray(data_shuffle)

#This gives a fixed block of data from a given start index.
def next_fixed_batch(num, data, startIndex):

    data_batch = data[:,startIndex:startIndex+num]    
    return np.asarray(data_batch)

'''
#total random columns of data with length num
def next_batch(num, data):
    
    #Return a total of `num` random samples and labels. 
    
    idx = np.arange(0 , len(data[0]))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[:,idx]

    return np.asarray(data_shuffle)
'''



def neural_network_model(data):
    output_layer = {'weights':tf.Variable(tf.random_normal([n_sources, n_sources])),
                    'biases':tf.Variable(tf.random_normal([n_sources, batch_size]))}
    net = tf.add(tf.matmul(output_layer['weights'],data), output_layer['biases'])
    output = tf.sigmoid(net)

    return output, output_layer['weights'],output_layer['biases']

def train_neural_network(x):
    information, W, Bias = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    #slice rows out of a 2d tensor
    Y1 = tf.slice(information,[0,0],[1,batch_size])
    Y2 = tf.slice(information,[1,0],[1,batch_size])
    costTotal = 0
    #Sums up the cost for all input vectors (2*1) in a batch
    for i in range(batch_size):
        #this accesses the ith element in a 1-d tensor
        y1 = Y1[0,i]
        y2 = Y2[0,i]
        costTotal += -tf.log(tf.abs(tf.matrix_determinant(W)*y1*(1-y1)*y2*(1-y2)))

    cost = costTotal/batch_size
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            #for _ in range(int(Ns/batch_size)):
            for _ in range(10):
                epoch_x= next_batch(batch_size,data)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        
        #a random matrix with shape 2*1
        Y = np.empty([2,1])
        startIndex = 0
        #for i in range(int(Ns/batch_size)):
        for i in range(100): #experiment
            new_batch = next_fixed_batch(batch_size, data, startIndex)
            Y_batch = sess.run(information, feed_dict={x: new_batch})
            Y = np.concatenate((Y,Y_batch), axis=1)
            startIndex += batch_size
        #deletes the initial random column
        np.delete(Y, 0, 1)

        #without adding back the mean
        # sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/info1.wav', Y[0,:], fs1)
        # sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/info2.wav', Y[1,:], fs1)
        
        #windows writing path
        sf.write('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\info1.wav', Y[0,:], fs1)
        sf.write('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\info2.wav', Y[1,:], fs1)


start_time = time.clock()

#train_neural_network(x)
cProfile.run('train_neural_network(x)')

print(time.clock() - start_time, "seconds")