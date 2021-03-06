import tensorflow as tf
import soundfile as sf
import numpy as np
import time
from tensorflow.python.client import timeline
import cProfile
from scipy.stats.stats import pearsonr   
import matplotlib
# matplotlib.use('TkAgg') 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

#This version of infomax uses the correlation to measure independence of real audio.
#It works with 2 source signals.

#read data, the type of data is a 1-D np.ndarray
data1, fs1 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig1.wav')
data2, fs2 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/a_sig2.wav')

#Windows reading path
# data1, fs1 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig1.wav')
# data2, fs2 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig2.wav')

#this sets the random seed to a fixed number.
np.random.seed(30)

n_sources = 2
batch_size = 10000
energy = 1
lowest_var = 0.01

#randomly initialize the mixing matrix A
#each entry is from uniform[0,1), 
A = np.random.rand(2,2)
invA = np.linalg.inv(A)

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

newA = np.dot(whiteningMatrix,A)

plt.figure()
plt.title('A')
AT = A.T
hor = AT[0,:]
ver = AT[1,:]
lineX1 = [0,hor[0]]
lineX2 = [0,hor[1]]
lineY1 = [0,ver[0]]
lineY2 = [0,ver[1]]
plt.plot(lineX1,lineY1)
plt.plot(lineX2,lineY2)
plt.savefig( "/home/yanlong/Downloads/2017T1/Comp489/ICA/plots/Infomax/figure1.png")

plt.figure()
plt.title('Whitened A')
newAT = newA.T
hor = newAT[0,:]
ver = newAT[1,:]
lineX1 = [0,hor[0]]
lineX2 = [0,hor[1]]
lineY1 = [0,ver[0]]
lineY2 = [0,ver[1]]
plt.plot(lineX1,lineY1)
plt.plot(lineX2,lineY2)
plt.savefig( "/home/yanlong/Downloads/2017T1/Comp489/ICA/plots/Infomax/figure2.png")

plt.figure()
plt.title('inverse Whitened A')
invnewAT = np.linalg.inv(newA.T)
hor = invnewAT[0,:]
ver = invnewAT[1,:]
lineX1 = [0,hor[0]]
lineX2 = [0,hor[1]]
lineY1 = [0,ver[0]]
lineY2 = [0,ver[1]]
plt.plot(lineX1,lineY1)
plt.plot(lineX2,lineY2)
plt.savefig( "/home/yanlong/Downloads/2017T1/Comp489/ICA/plots/Infomax/figure6.png")

plt.figure()
plt.title('inverse A')
invAT = invA.T
hor = invAT[0,:]
ver = invAT[1,:]
lineX1 = [0,hor[0]]
lineX2 = [0,hor[1]]
lineY1 = [0,ver[0]]
lineY2 = [0,ver[1]]
plt.plot(lineX1,lineY1)
plt.plot(lineX2,lineY2)
plt.savefig( "/home/yanlong/Downloads/2017T1/Comp489/ICA/plots/Infomax/figure3.png")

data = whitened
data = np.transpose(data)

# data = V.T

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
    # C = tf.ones([n_sources, n_sources])
    # average = energy*1.0/n_sources
    # C = tf.multiply(average,C)
    output_layer = {'weights':tf.Variable(tf.random_normal([n_sources, n_sources])),
                    'biases':tf.Variable(tf.random_normal([n_sources]))}
    # output_layer = {'weights':tf.Variable(C),
    #             'biases':tf.Variable(tf.random_normal([n_sources]))}
    net = tf.nn.bias_add(tf.matmul(data,output_layer['weights']), output_layer['biases'])
    output = tf.sigmoid(net)
    # output = net
    pure = net
   
    return output, output_layer['weights'],output_layer['biases'], pure

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
    var_diff = 0
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
    var_diff += 10*tf.nn.relu(lowest_var - variance1)
    var_diff += 10*tf.nn.relu(lowest_var - variance2)
    # divisor = tf.sqrt(
    #     (tf.reduce_mean(tf.square(Y1)) - tf.square(tf.reduce_mean(Y1) ) ) *
    #     (tf.reduce_mean(tf.square(Y2)) - tf.square(tf.reduce_mean(Y2) ) )
    #                 )
    original_loss = tf.truediv(numerator, divisor)
    ortho = tf.abs(tf.reduce_sum(tf.multiply(W[:,0], W[:,1]) ) )
    absW = tf.abs(W)
    # quan = tf.abs(tf.reduce_sum(absW[:,0]) - tf.reduce_sum(absW[:,1]))
    quan = tf.abs(tf.reduce_sum(absW[:,0]) - energy)
    quan += tf.abs(tf.reduce_sum(absW[:,1]) - energy)

    #cost = -tf.log(1-tf.abs(original_loss)+epsilon)
    #cost = tf.abs(original_loss)#+0.1*tf.abs(1-variance1)+0.1*tf.abs(1-variance2)
    cost = tf.abs(original_loss)+ortho+quan+var_diff
    #TODO I need to keep the variance constant.
    return cost, variance1, variance2, numerator


def train_neural_network(x):
    information, W, Bias, pure = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:

    #dynamic learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           11200, 0.1, staircase=True)
    
    # learning_rate = 0.0001
    cost, v1, v2, num = calculate_cost(information,W)
    #Add learning rate 1e-5
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)
    
    hm_epochs = 20000

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
        anchor = 100
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
                # if step % 1 == 0:
                #     print('Epoch', epoch, 'cost', c)
                #     print('var1:',var1)
                #     print('var2:',var2)
                #     print('nume:',nume)
                #     print('weights',weights)

            epoch_loss = epoch_loss/(int(Ns/batch_size))
            if epoch_loss < 0.02 and epoch > 500: break
            if epoch % 200 == 0:
                if anchor > epoch_loss:
                    anchor = epoch_loss
                else:
                    break
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print(weights)

        print('A:',A.T)
        print('invA:',invA.T)
        print('whitened A:',newA.T)

        plt.figure()
        plt.title('Rotation')
        hor = weights[0,:]
        ver = weights[1,:]
        lineX1 = [0,hor[0]]
        lineX2 = [0,hor[1]]
        lineY1 = [0,ver[0]]
        lineY2 = [0,ver[1]]
        plt.plot(lineX1,lineY1)
        plt.plot(lineX2,lineY2)
        plt.savefig( "/home/yanlong/Downloads/2017T1/Comp489/ICA/plots/Infomax/figure4.png")

        #Y = sess.run(information, feed_dict={x: data}, options=run_options, run_metadata=run_metadata)
        Y = sess.run(information, feed_dict={x: data})
        P = sess.run(pure, feed_dict={x: data})

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

        #now do the scaling for P to minimize sum of squared error
        value1 = np.dot(P.T,S.T)
        value2 = np.dot(P.T,P)
        value3 = np.linalg.inv(value2)
        rho = np.dot(value3,value1)
        P = np.dot(P,rho)

        print('magic:')
        print(np.dot(weights.T,newA))

        plt.figure()
        plt.title("signal plot")
        ax1 = plt.subplot(611)
        plt.plot(data1[:Ns*2])
        ax2 = plt.subplot(612, sharex=ax1)
        plt.plot(data2[:Ns*2])
        ax3 = plt.subplot(613, sharex=ax1)
        plt.plot(Y.T[0,:Ns*2])
        ax4 = plt.subplot(614, sharex=ax1)
        plt.plot(Y.T[1,:Ns*2])
        ax5 = plt.subplot(615, sharex=ax1)
        plt.plot(P.T[0,:Ns*2])
        ax6 = plt.subplot(616, sharex=ax1)
        plt.plot(P.T[1,:Ns*2])
        plt.savefig( "/home/yanlong/Downloads/2017T1/Comp489/ICA/plots/Infomax/figure5.png")

        #without adding back the mean
        # sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/cor1.wav', Y[:,0], fs1)
        # sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/cor2.wav', Y[:,1], fs1)

        # sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/corP1.wav', P[:,0], fs1)
        # sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/corP2.wav', P[:,1], fs1)

        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/cor1.wav', Y.T[0,:], fs1)
        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/cor2.wav', Y.T[1,:], fs1)

        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/corP1.wav', P.T[0,:], fs1)
        sf.write('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/corP2.wav', P.T[1,:], fs1)

        return epoch_loss, P
    
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
# cProfile.run('train_neural_network(x)')

loss, V = train_neural_network(x)

while loss > 0.17:
    V = V.T
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

    newA = np.dot(weights.T,newA)

    loss, V = train_neural_network(x)

#print(time.clock() - start_time, "seconds")
