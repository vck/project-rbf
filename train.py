import numpy as np  
import pandas as pd
import math as m
from random import shuffle
from math import pow
from time import time
from newrbf import kmeans, distance



"""
fungsi pelatihan
----------------

args:
	- x: data latih 
	- y: data latih target
	- n_hidden: banyak hidden layer
	- ephocs: banyak perulangan pelatihan
	- lr: learning rate

returns:
	- (N, 1) weigh
"""

# parameter
x = np.linspace(0, 1, num=10)
y = np.sin(x*np.pi)
#df = pd.read_excel("../Data/Data Hasil Panen Normalisasii.xlsx", skiprows=9)

#columns = ["PROUCTION", "POKOK PANEN"]

#x = df[columns[0]].values
#y = df[columns[1]].values

n_hidden = 3
ephocs=500
lr=0.0000000001
N = x.shape[0]
initial_weight = [0 for i in range(n_hidden)]
initial_weight.append(0)
initial_weight = np.array(initial_weight)
# tambahkan bias

k = n_hidden

center_dist = kmeans(x, k)

center = [i[1] for i in center_dist]
distance = [i[0] for i in center_dist]

betas = [distance[i]/m.sqrt(center[i]) for i in range(len(center))]
# beta di hidden 1, 2, 3,..., N

outputs = []

for param in range(len(betas)):
	cluster_input = sum([m.exp((x_i-center[param])**2) for x_i in x])
	output = cluster_input / (2 * betas[param] ** 2)
	outputs.append(output)

outputs.append(1.0) # tambahkan bias pada layer terakhir
# shape output = (n_hidden+1, 1)

outputs = np.array(outputs)
n = 0
run_training = True
for ephoc in range(ephocs):
	# training
	for y_ in y:

		out = np.dot(outputs, initial_weight)
		error = y_ - out

		delta_w = error * outputs
		w_update = initial_weight + lr * delta_w
		initial_weight = w_update

		print("error: {} data: {}".format(error, y_	))
	
	print("ephoc: %s"%ephoc)
	
print(initial_weight)
