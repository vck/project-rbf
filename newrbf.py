import numpy as np  
import pandas as pd
import math as m
from random import shuffle
from math import pow
from time import time


def distance(a, y):
	error = a-y
	dist = m.sqrt((error)**2)
	return dist

def calculate_distance(x_j, c_j):
	"""
	jarak antara x di hidden j dengan center di hidden j  

	args:
		- x: scalar (int, float)
		- c_j: center pada hidden j (int, float)
		 
	"""
	return np.square(pow(x_j-c_j, 2)) # akar kuadrat dari selisih antara x_j dan c_j


def calculate_error(t_k, y_k):
	"""
	Beda antara data target dengan prediksi model

	args:
		- t_k: target pada hidden layer k 
		- y_k: hasil prediksi pada hidden layer k
	"""
	return t_k - y_k 


def calculate_betha_(d_max, c_j):
	"""Hitung nilai beta

	args: 
		- dmax: data terbesar 
		  c_j: center pada hidden layer j
	
	returns:
	    float
	"""
	return d_max / np.sqrt(c_j)


def preprocessing(x):
	"""
	tahap pra pemrosesan, normalisasi data

	args: 
		x: n-dimentional vector [a..b]
	returns:
		x: normalized x n-dimentional vector
	
	x_final = (d-d_min) / (d_max-d_min)
	"""
	dmax = max(x)
	dmin = min(x)
	x_normalized = [(x_-dmin)/(dmax-dmin) for x_ in x]
	return x_normalized


def denormalisasi(x):
	pass


def kmeans(x, k):
	k = 3
	init_center = x[:k]
	tidak_sama = True 
	counter = 0
	t0 = time()

	while tidak_sama:
		center_candidate = []

		for x_ in x:
			dist_data = {}
			for center in init_center:
				dist = distance(x_, center)
				dist_data[dist] = (x_, center)

			center_candidate.append(dist_data)
		
		nc = []
		# 1/ N * sum(X1+X2+X3+...+XN)
		dmaxes = []

		for i in center_candidate:
			min_ = min(i.keys())
			nc.append(i[min_])
			#dmaxes.append(max(i.keys()))

		hadeeh = {}
		hadeeeeh = {}

		for hani_jelek in nc:
	    	# bikin key dulu sama list kosong bosqu hadeeeh
			hadeeh[hani_jelek[1]] =[]
		
		for hani_jelek in nc:
			for key in hadeeh:
				if hani_jelek[1] == key:
					hadeeh[key].append(hani_jelek[0])

		new_center = []

		for k in hadeeh.keys():
			N = len(hadeeh[k])
			c_new = 1/N * sum(hadeeh[k])
			new_center.append(c_new)

		kandidat_jarak = []
		for center in new_center:
			for candidate in center_candidate:
				prev_max = 0
				dmax = max(candidate.keys())
				if dmax > prev_max:
					prev_max = dmax
					kandidat_jarak.append((prev_max, center))
					break

		if np.array_equal(init_center, new_center):
			tidak_sama = False
		else:
			counter = counter+1
			init_center = new_center

	return kandidat_jarak


def train(x, y, n_hidden, ephocs=500, lr=0.00001):
	
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

	N = x.shape[0]
	initial_weight = [0 for i in range(n_hidden)]
	initial_weight.append(1)
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


	for i in range(ephocs):
		for x_i in range(N):
			out = outputs * initial_weight
			error = y[x_i] - out
			delta_w = error * outputs
			if i%10 == 0:
				print("ephocs: {} error: {}".format(i, error))
			w_update = initial_weight + lr * delta_w
			if np.array_equal(w_update, initial_weight):
				print("optimum weight found")
				print(w_update)
				break
			initial_weight = w_update

	return initial_weight
	
if __name__ == '__main__':
	#x = np.linspace(0, 1, num=100)
	df = pd.read_excel("../Data/Data Hasil Panen Normalisasii.xlsx", skiprows=9)
	
	columns = ["PROUCTION", "POKOK PANEN"]
	
	x = df[columns[0]].values
	y = df[columns[1]].values

	weight = train(x, y, 3)



