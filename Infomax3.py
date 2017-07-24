import tensorflow as tf
import soundfile as sf
import numpy as np
import time
from tensorflow.python.client import timeline
import cProfile
from scipy.stats.stats import pearsonr   

#This version of infomax uses the correlation to measure independence of real audio.

#read data, the type of data is a 1-D np.ndarray
data1, fs1 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig1.wav')
data2, fs2 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig2.wav')

#Windows reading path
# data1, fs1 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig1.wav')
# data2, fs2 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig2.wav')

#this sets the random seed to a fixed number.
np.random.seed(10)

n_sources = 2
batch_size = 1000

#randomly initialize the mixing matrix A
#each entry is from uniform[0,1), 
A = np.random.rand(2,2)

#the number of data points. Also the number of columns.
#Ns = len(data1)
Ns = fs1 * 7 #self defined data length, 5 seconds of speech
data1 = data1[:Ns]
data2 = data2[:Ns]

print('Ns',Ns)

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
#     epsilon = 1e-8
#     # original_loss =  (tf.reduce_sum(tf.matmul(tf.transpose(Y1), Y2)) - (tf.reduce_sum(Y1) * tf.reduce_sum(Y2)))*1.0/batch_size
#     # divisor = tf.sqrt(
#     #     (tf.reduce_sum(tf.square(Y1))*1.0/batch_size - tf.square(tf.reduce_sum(Y1)*1.0/batch_size ) ) *
#     #     (tf.reduce_sum(tf.square(Y2))*1.0/batch_size - tf.square(tf.reduce_sum(Y2)*1.0/batch_size ) )
#     #                 )

#     original_loss =  tf.reduce_mean(tf.matmul(tf.transpose(Y1), Y2))*1.0/batch_size - (tf.reduce_mean(Y1) * tf.reduce_mean(Y2)) 
#     variance1 = tf.reduce_mean(tf.square(Y1)) - tf.square(tf.reduce_mean(Y1) )
#     variance2 = tf.reduce_mean(tf.square(Y2)) - tf.square(tf.reduce_mean(Y2) )
#     divisor = tf.sqrt(variance1*variance2)
#     # divisor = tf.sqrt(
#     #     (tf.reduce_mean(tf.square(Y1)) - tf.square(tf.reduce_mean(Y1) ) ) *
#     #     (tf.reduce_mean(tf.square(Y2)) - tf.square(tf.reduce_mean(Y2) ) )
#     #                 )
#     original_loss = tf.truediv(original_loss, divisor)
#     #cost = -tf.log(1-tf.abs(original_loss)+epsilon)
#     cost = tf.abs(original_loss)+epsilon#+tf.abs(variance1-variance2)
#     #TODO I need to keep the variance constant.
#     return cost

def calculate_cost(unmixed,W):
    #slice rows out of a 2d tensor
    Y1 = tf.slice(unmixed,[0,0],[batch_size,1])
    Y2 = tf.slice(unmixed,[0,1],[batch_size,1])
    epsilon = 1e-8
    # original_loss =  (tf.reduce_sum(tf.matmul(tf.transpose(Y1), Y2)) - (tf.reduce_sum(Y1) * tf.reduce_sum(Y2)))*1.0/batch_size
    # divisor = tf.sqrt(
    #     (tf.reduce_sum(tf.square(Y1))*1.0/batch_size - tf.square(tf.reduce_sum(Y1)*1.0/batch_size ) ) *
    #     (tf.reduce_sum(tf.square(Y2))*1.0/batch_size - tf.square(tf.reduce_sum(Y2)*1.0/batch_size ) )
    #                 )

    # numerator =  tf.reduce_mean(tf.matmul(tf.transpose(Y1), Y2))*1.0/batch_size - (tf.reduce_mean(Y1) * tf.reduce_mean(Y2))
    numerator =  tf.reduce_mean(tf.multiply(Y1, Y2)) - (tf.reduce_mean(Y1) * tf.reduce_mean(Y2))
    variance1 = tf.reduce_mean(tf.square(Y1)) - tf.square(tf.reduce_mean(Y1) ) + epsilon
    variance2 = tf.reduce_mean(tf.square(Y2)) - tf.square(tf.reduce_mean(Y2) ) + epsilon
    #avoid the small negative number issue
    numerator = tf.nn.relu(numerator)
    variance1 = tf.nn.relu(variance1)+epsilon
    variance2 = tf.nn.relu(variance2)+epsilon
    divisor = tf.sqrt(variance1*variance2)+epsilon
    # divisor = tf.sqrt(
    #     (tf.reduce_mean(tf.square(Y1)) - tf.square(tf.reduce_mean(Y1) ) ) *
    #     (tf.reduce_mean(tf.square(Y2)) - tf.square(tf.reduce_mean(Y2) ) )
    #                 )
    original_loss = tf.truediv(numerator, divisor)
    #cost = -tf.log(1-tf.abs(original_loss)+epsilon)
    cost = tf.abs(original_loss)#+0.1*tf.abs(1-variance1)+0.1*tf.abs(1-variance2)
    #TODO I need to keep the variance constant.
    return cost, variance1, variance2, numerator


def train_neural_network(x):
    information, W, Bias = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    
    cost, v1, v2, num = calculate_cost(information,W)
    #Add learning rate 1e-5
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)
    
    hm_epochs = 300

    #try to disable all the gpus
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:

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
                _, c, weights, var1, var2, nume = sess.run([optimizer, cost, W, v1, v2, num], feed_dict={x: data[startIndex:startIndex+batch_size,:]})
                # _, c, weights = sess.run([optimizer, cost, W], feed_dict={x: data[startIndex:startIndex+batch_size,:]})
                epoch_loss += c
                # The following prints the intermediate steps in each epoch
                step+=1
                if step % 1 == 0:
                    print('Epoch', epoch, 'cost', c)
                    print('var1:',var1)
                    print('var2:',var2)
                    print('nume:',nume)
                    print('weights',weights)

            epoch_loss = epoch_loss/(int(Ns/batch_size))
            if epoch_loss == 0: break
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print(weights)

        #Y = sess.run(information, feed_dict={x: data}, options=run_options, run_metadata=run_metadata)
        Y = sess.run(information, feed_dict={x: data})

        #without adding back the mean
        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/cor1.wav', Y[:,0], fs1)
        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/cor2.wav', Y[:,1], fs1)
    
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