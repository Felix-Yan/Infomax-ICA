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

data, fs1 = sf.read('/home/yanlong/Downloads/2017T1/Comp489/ICA/Data/cor1.wav')
var = np.var(data)
print(var)
plt.plot(data)
plt.show()
