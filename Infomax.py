import tensorflow as tf
import soundfile as sf
import numpy as np

#read data, the type of data is a 1-D np.ndarray
data1, fs1 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig1.wav')
data2, fs2 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig2.wav')

#this sets the random seed to a fixed number.
np.random.seed(10)

#randomly initialize the mixing matrix A
#each entry is from uniform[0,1), 
A = np.random.rand(2,2)
#stack the two data arrays together as the source signals
#the shape of S is (2,Ns)
S = np.array((data1,data2))

#the number of data points. Also the number of columns.
Ns = len(data1)
#V is the observed signal mixture.
V = np.dot(A,S)

data = V

n_sources = 2
batch_size = 100

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

    return output, output_layer['weights']

def train_neural_network(x):
    information, W = neural_network_model(x)
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

train_neural_network(x)