import numpy as np
from scipy.stats import ortho_group
import itertools
from scipy.stats.stats import pearsonr   

def synthesize(Ns,data_size):

	#this sets the random seed to a fixed number.
	np.random.seed(10)
	#data_size = 45
	p = 2 #number of signals in one input
	#this sets the mean(loc) and scale for a laplacian distribution
	loc, scale = 0., 1.
	shape = 2.
	#the following lists hold the independent signals 
	training1 = []
	training2 = []
	testing1 = []
	testing2 = []
	for i in range(data_size):
		s1 = np.random.laplace(loc, scale, Ns)
		s2 = np.random.gamma(shape, scale, Ns)
		# s2 = np.random.laplace(loc, scale, Ns)
		s3 = np.random.laplace(loc, scale, Ns)
		s4 = np.random.gamma(shape, scale, Ns)
		# s4 = np.random.laplace(loc, scale, Ns)
		training1.append(s1)
		training2.append(s2)
		testing1.append(s3)
		testing2.append(s4)
	#the following lists hold the pairs of independent signals
	training_subsets1 = []
	training_subsets2 = []
	testing_subsets1 = []
	testing_subsets2 = []
	#141 choose 2, no order, no repetition. there should be 10011 subsets here.
	#45 choose 2 is 990.
	for subset in itertools.combinations(training1, p):
	    training_subsets1.append(subset)
	for subset in itertools.combinations(training2, p):
	    training_subsets2.append(subset)
	for subset in itertools.combinations(testing1, p):
	    testing_subsets1.append(subset)
	for subset in itertools.combinations(testing2, p):
	    testing_subsets2.append(subset)


	number = len(training_subsets1)
	training_Mixed = []
	training_Unmixed = []
	testing_Mixed = []
	testing_Unmixed = []
	#The following lists hold the correlations of each pair in the precious lists.
	cor_list1 = []
	cor_list2 = []
	cor_list3 = []
	cor_list4 = []
	mixing_matrices1 = []
	mixing_matrices2 = []
	#concatenate signals. Mix signals.
	for i in range(number):
		#A1 = ortho_group.rvs(dim=p) #this creates a orthonormal matrix
		#A2 = ortho_group.rvs(dim=p)
		A1 = np.random.rand(p,p)
		A2 = np.random.rand(p,p)
		mixing_matrices1.append(A1)
		mixing_matrices2.append(A2) 
		pair1 = training_subsets1[i]
		data11 = pair1[0]
		data12 = pair1[1]
		pair2 = training_subsets2[i]
		data21 = pair2[0]
		data22 = pair2[1]
		pair3 = testing_subsets1[i]
		data31 = pair3[0]
		data32 = pair3[1]
		pair4 = testing_subsets2[i]
		data41 = pair4[0]
		data42 = pair4[1]
		#stack the two data arrays together as the source signals
		#the shape of S is (2,Ns)
		S1 = np.array((data11,data12))
		S2 = np.array((data21,data22))
		S3 = np.array((data31,data32))
		S4 = np.array((data41,data42))
		#V is the observed signal mixture.
		V1 = np.dot(A1,S1)
		V2 = np.dot(A2,S3)

		cor1 = pearsonr(V1[0,:],V1[1,:])[0]
		cor2 = pearsonr(data21,data22)[0]
		cor3 = pearsonr(V2[0,:],V2[1,:])[0]
		cor4 = pearsonr(data41,data42)[0]

		training_Mixed.append(V1)
		training_Unmixed.append(S2)
		testing_Mixed.append(V2)
		testing_Unmixed.append(S4)

		cor_list1.append(cor1)
		cor_list2.append(cor2)
		cor_list3.append(cor3)
		cor_list4.append(cor4)

	#The following lists hold the preprocessed input data
	training_Mixed2 = []
	training_Unmixed2 = []
	testing_Mixed2 = []
	testing_Unmixed2 = []
	for i in range(number):
		data1 = preprocessing(training_Mixed[i],Ns,p)
		data2 = preprocessing(training_Unmixed[i],Ns,p)
		data3 = preprocessing(testing_Mixed[i],Ns,p)
		data4 = preprocessing(testing_Unmixed[i],Ns,p)
		training_Mixed2.append(data1)
		training_Unmixed2.append(data2)
		testing_Mixed2.append(data3)
		testing_Unmixed2.append(data4)

	cor_lists = [cor_list1,cor_list2,cor_list3,cor_list4]
	return training_Mixed2, training_Unmixed2, testing_Mixed2, testing_Unmixed2, cor_lists, mixing_matrices1, mixing_matrices2

#Mean subtraction. Whitening. V is data matrix.
def preprocessing(V,Ns,n_sources):
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

	return data