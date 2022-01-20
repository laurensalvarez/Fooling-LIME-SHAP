#imports all functions from modules without haveing to call module. before
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

# This module provides generic shallow and deep copy operations (explained below).
from copy import deepcopy

#given a json_path str, returns parameter object
#includes def one_hot_encode(y), rank_features(explanation), get_rank_map(ranks, to_consider), experiment_summary(explanations, features)
params = Params("model_configurations/experiment_params.json")
# returns Pandas data frame (Two-dimensional, size-mutable, potentially heterogeneous tabular data.) X of processed data, np.ndarray y, and list of column names
X, y, cols = get_and_preprocess_compas_data(params)
# list comprehension for features = the column names
features = [c for c in X]

race_indc = features.index('race')

#Returns a Numpy rep of the DF. only the vlaues will be returned. axes labels are removed
X = X.values
#list of column indexes
c_cols = [features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")]

#Compute the mean and std to be used for later scaling.
ss = StandardScaler().fit(X)
#Perform standardization by centering and scaling.
X = ss.transform(X)

#instantiate list r
r = []
# _ = Ignore the index
for _ in range(3):
    # returns Drawn samples from the parameterized normal distribution.
	p = np.random.normal(0,1,size=X.shape)

    # X pertubated?
	X_p = X + p
    # : is a slice. you have to slice to make a copy instead of changing the whole thing
    # it's only copying the column names
	X_p[:, c_cols] = X[:, c_cols]

	for row in X_p:
		for c in c_cols:
            # Generates a random sample from a given 1-D array
			row[c] = np.random.choice(X_p[:,c])

	r.append(X_p)

r = np.vstack(r)
p = [1 for _ in range(len(r))]
iid = [0 for _ in range(len(X))]

all_x = np.vstack((r,X))
all_y = np.array(p + iid)

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
results = pca.fit_transform(all_x)

print (len(X))

plt.scatter(results[:2000,0], results[:2000,1], alpha=.1)
plt.scatter(results[-int(X.shape[0]*.10):,0], results[-int(X.shape[0]*.10):,1], alpha=.1)
plt.show()
