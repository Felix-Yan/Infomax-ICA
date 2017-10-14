import tensorflow as tf
import soundfile as sf
import numpy as np
import time
from tensorflow.python.client import timeline
import cProfile

#read data, the type of data is a 1-D np.ndarray
data1, fs1 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig1.wav')
data2, fs2 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig2.wav')
data3, fs3 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig3.wav')

#Windows reading path
# data1, fs1 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig1.wav')
# data2, fs2 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig2.wav')

#this sets the random seed to a fixed number.
np.random.seed(10)

n_sources = 3
batch_size = 100

#randomly initialize the mixing matrix A
#each entry is from uniform[0,1), 
A = np.random.rand(n_sources,n_sources)

#the number of data points. Also the number of columns.
#Ns = len(data1)
Ns = fs1 * 7 #self defined data length, 5 seconds of speech
data1 = data1[:Ns]
data2 = data2[:Ns]
data3 = data3[:Ns]

#stack the two data arrays together as the source signals
#the shape of S is (2,Ns)
S = np.array((data1,data2,data3))
# S = np.array((data1,data2))

#V is the observed signal mixture.
V = np.dot(A,S)
# V = S

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
data = np.transpose(data)

#None means it can be any value
x = tf.placeholder('float', [None, n_sources])


#The two functions below are not necessary 
#This give s a random block of data with size num
def next_batch(num, data):

    #Return a total of `num` random samples and labels. 

    idx = np.arange(0 , len(data)-num)
    np.random.shuffle(idx)
    idx = idx[0]
    #This gives num random columns of the data array
    data_shuffle = data[idx:idx+num,:]

    return np.asarray(data_shuffle)

#This gives a fixed block of data from a given start index.
def next_fixed_batch(num, data, startIndex):

    data_batch = data[startIndex:startIndex+num,:]    
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
                    'biases':tf.Variable(tf.random_normal([n_sources]))}
    net = tf.nn.bias_add(tf.matmul(data,output_layer['weights']), output_layer['biases'])
    output = tf.sigmoid(net)
   
    return output, output_layer['weights'],output_layer['biases']

# def calculate_cost(unmixed,W):
#     #slice rows out of a 2d tensor
#     Y1 = tf.slice(unmixed,[0,0],[batch_size,1])
#     Y2 = tf.slice(unmixed,[0,1],[batch_size,1])
#     costTotal = 0
#     epsilon = 1e-8
#     #Sums up the cost for all input vectors (2*1) in a batch
#     for i in range(batch_size):
#         #this accesses the ith element in a 1-d tensor
#         y1 = Y1[i,0]
#         y2 = Y2[i,0]
#         #costTotal += -tf.log(tf.abs(tf.matrix_determinant(W+np.identity(2)*epsilon)*y1*(1-y1)*y2*(1-y2)))
#         #mat_deter = tf.matrix_determinant(W+tf.to_float(np.identity(2))*epsilon)
#         mat_deter = tf.matrix_determinant(W)
#         #costTotal += -tf.log(tf.abs(mat_deter)*y1*(1-y1)*y2*(1-y2)+epsilon)+0.01*tf.norm(W, ord='fro', axis=[0,1])
#         costTotal += -tf.log(tf.abs(mat_deter)*y1*(1-y1)*y2*(1-y2)+epsilon)


#     cost = costTotal/batch_size
#     return cost


def calculate_cost(unmixed,W):
    epsilon = 1e-8

    #slice rows out of a 2d tensor
    Y1 = tf.slice(unmixed,[0,0],[batch_size,1])
    Y2 = tf.slice(unmixed,[0,1],[batch_size,1])
    Y3 = tf.slice(unmixed,[0,2],[batch_size,1])
    Y1P = tf.subtract(1.,Y1)
    Y2P = tf.subtract(1.,Y2)
    Y3P = tf.subtract(1.,Y3)
    #add epsilon here
    Y1 = tf.add(epsilon,Y1)
    Y2 = tf.add(epsilon,Y2)
    Y3 = tf.add(epsilon,Y3)

    Y1P = tf.add(epsilon,Y1P)
    Y2P = tf.add(epsilon,Y2P)
    Y3P = tf.add(epsilon,Y3P)

    Y1 = tf.log(Y1)
    Y2 = tf.log(Y2)
    Y3 = tf.log(Y3)

    Y1P = tf.log(Y1P)
    Y2P = tf.log(Y2P)
    Y3P = tf.log(Y3P)

    y1 = tf.reduce_mean(Y1)
    y2 = tf.reduce_mean(Y2)
    y3 = tf.reduce_mean(Y3)

    y1p = tf.reduce_mean(Y1P)
    y2p = tf.reduce_mean(Y2P)
    y3p = tf.reduce_mean(Y3P)

    mat_deter = tf.matrix_determinant(W)

    cost = -(tf.log(tf.abs(mat_deter)+epsilon) + y1 + y2 + y3 + y1p + y2p + y3p)
    # cost = -(tf.log(tf.abs(mat_deter)+epsilon) + y1 + y2 + y1p + y2p )
    return cost


def train_neural_network(x):
    information, W, Bias = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    
    cost = calculate_cost(information,W)
    #Add learning rate 1e-5
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)
    
    hm_epochs = 60

    #try to disable all the gpus
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    with tf.Session(config=config) as sess:
    # with tf.Session() as sess:

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())

        #this prints out the training variables in tensorflow
        tvars=tf.trainable_variables()
        myvars = sess.run(tvars)
        print(myvars)
        # sess.close()

        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
    
        for epoch in range(hm_epochs):
            epoch_loss = 0
            step = 0
            for _ in range(int(Ns/batch_size)):
                #epoch_x= next_batch(batch_size,data)
                startIndex = step * batch_size
                # _, c, det = sess.run([optimizer, cost, mat_deter], feed_dict={x: epoch_x}, options=run_options, run_metadata=run_metadata)
                #_, c, det = sess.run([optimizer, cost, mat_deter], feed_dict={x: epoch_x})
                _, c, weights = sess.run([optimizer, cost, W], feed_dict={x: data[startIndex:startIndex+batch_size,:]})
                epoch_loss += c
                # The following prints the intermediate steps in each epoch
                step+=1
                # if step % 50 ==0:
                #     print('Epoch', epoch, 'cost', c)

            epoch_loss = epoch_loss/(int(Ns/batch_size))
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print(weights)

        #Y = sess.run(information, feed_dict={x: data}, options=run_options, run_metadata=run_metadata)
        Y = sess.run(information, feed_dict={x: data})

        Y = np.transpose(Y)
        meanValueY = np.mean(Y, axis = 1)
        #This changes meanValue from 1d to 2d, now a column vector with size dimension*1
        meanValueY = np.reshape(meanValueY,(len(meanValueY),1))
        #This creates an array full of ones with the same length as the column number of V
        oneArrayY = np.ones((1,Ns))
        #This creates a matrix full of mean values for each row
        meanMatrixY = np.dot(meanValueY,oneArrayY)
        #This gives V zero mean
        Y = Y - meanMatrixY
        Y = np.transpose(Y)

        #without adding back the mean
        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/info1.wav', Y[:,0], fs1)
        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/info2.wav', Y[:,1], fs1)
        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/info3.wav', Y[:,2], fs1)
    
        #windows writing path
        # sf.write('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\info1.wav', Y[:,0], fs1)
        # sf.write('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\info2.wav', Y[:,1], fs1)

        #Create the Timeline object, and write it to a json
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(ctf)


#start_time = time.clock()

#train_neural_network(x)
cProfile.run('train_neural_network(x)')

#print(time.clock() - start_time, "seconds")

# 2 inputs weight matrix
# [[ 1.38245511 -1.91294873]
#  [-2.0170188  -1.30917716]]



# 3 inputs weights
# [[-0.97527641 -1.20582628 -1.73407209]
 # [ 0.79302579 -2.09988451  0.82884341]
 # [ 1.81829226  0.26965806 -1.29209054]]


