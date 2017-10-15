# import tensorflow as tf
import soundfile as sf
import numpy as np
import time
# from tensorflow.python.client import timeline
import cProfile
from scipy.stats.stats import pearsonr   
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

np.random.seed(10)

A = np.random.rand(2,2)
print(A.T)

plt.plot(A.T)

plt.show()