#imports all functions from modules without haveing to call module.() before
from adversarial_models import *
from utils import *
from get_data import *


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

#this module provides generic shallow and deep copy operations (explained below).
from copy import deepcopy

#given a json_path str, returns parameter object
#includes def one_hot_encode(y), rank_features(explanation), get_rank_map(ranks, to_consider), experiment_summary(explanations, features)
params = Params("model_configurations/experiment_params.json")
#returns Pandas data frame (Two-dimensional, size-mutable, potentially heterogeneous tabular data.) X of processed data, np.ndarray y, and list of column names
X, y, cols = get_and_preprocess_compas_data(params)
#list comprehension for features = the column names
features = [c for c in X]

race_indc = features.index('race')

#returns a Numpy rep of the DF. only the vlaues will be returned. axes labels are removed
X = X.values
#list of column indexes
c_cols = [features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")]

#compute the mean and std to be used for later scaling.
ss = StandardScaler().fit(X)
#perform standardization by centering and scaling.
X = ss.transform(X)

#instantiate list r ... perhaps the perturbated data??
r = []
#_ = ignore the index
for _ in range(3):
    #returns drawn samples from the parameterized normal distribution.
	p = np.random.normal(0,1,size=X.shape)

    #X perturbated?
	X_p = X + p
    #: is a slice. you have to slice to make a copy instead of changing the whole thing
    #it's only copying the column names
	X_p[:, c_cols] = X[:, c_cols]

    #for all the rows in X_p create
	for row in X_p:
		for c in c_cols:
            #generates a random sample from a X_p at that column
			row[c] = np.random.choice(X_p[:,c])

	r.append(X_p)
#i want to know what X_p & r looks like
#print('this is what r looks like:')
#print(r)

#stacks arrays in sequence vertically, makes it 2D??
r = np.vstack(r)
print("r.shape")
print(r.shape)
# 1 is the perturbed class & 0 is the OG class
#p = pertubated points & how far they move it... put a 1 for each var in the range of length of r
p = [1 for _ in range(len(r))]
#iid = independent variables with identical distributions & put a 0 for each var in the range of length of X
iid = [0 for _ in range(len(X))]

#stacks to pertubations and the original standardized numpy X
all_x = np.vstack((r,X))
#creates an array of perturbed points & points with the same probability distribution
#--> use for classifier; check to see if it's right
all_y = np.array(p + iid)

print("all x data", all_x)
print("--------------------------------------")
print("all y data", all_y)

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
#fit the model with all_x and apply the dimensionality reduction on all_x. returns ndarray of shape(n_samples, n_compnents)
results = pca.fit_transform(all_x)
print("results.shape")
print(results.shape)
print("X.shape")
print(X.shape)
print("all_y.shape")
print(all_y.shape)
print("length of X")
print (len(X))


#BLUE scatter plot( x = results[items from the beginning through 2000/stop-1 , column_0], y = results[items from the beginning through 2000/stop-1 , column_1], blending value 0-1)
# plt.scatter(results[:2000,0], results[:2000,1], alpha=.1)
# #ORANGE scatter plot( x = results[idk but items start through the rest of the array , column_0], y = results[idk , column_1], blending value 0-1)
# plt.scatter(results[-int(X.shape[0]*.10):,0], results[-int(X.shape[0]*.10):,1], alpha=.1)
# plt.show()
