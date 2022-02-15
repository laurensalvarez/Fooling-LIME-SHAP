
# Adapted from: https://github.com/iamsiter/FastMap-algorithm-in-Python/blob/master/FastMapUpdated.ipynb

# ------------------------------------------------------------------------------
# SCRUBBING TOGETHER COMPAS DATA
# ------------------------------------------------------------------------------

from utils import *
from get_data import *

#1 Importing the libraries
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import pandas as pd
import math

import lime
import lime.lime_tabular
import shap
from copy import deepcopy

from sklearn.preprocessing import StandardScaler



#2 Importing the COMPAS dataset
# Size = 6172
# Features: criminal history, demographics, COMPAS risk score, jail and prison time
# Positive class: High Risk (81.4%)
# Sensitive Feature: African-American (51.4%)
params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_compas_data(params)
features = [c for c in X]
race_indc = features.index('race')

X = X.values
#list of column indexes
c_cols = [features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")]

#2b Feature Scaling
#compute the mean and std to be used for later scaling.
ss = StandardScaler().fit(X)
#perform standardization by centering and scaling.
X = ss.transform(X)

#3 creating the pertubations from lime manually
lime_points = []
for _ in range(3):
	p = np.random.normal(0,1,size=X.shape)

	X_p = X + p
	X_p[:, c_cols] = X[:, c_cols]

	for row in X_p:
		for c in c_cols:
			row[c] = np.random.choice(X_p[:,c])

	lime_points.append(X_p)

lime_points = np.vstack(lime_points)
# 1 is the perturbed class & 0 is the OG class
#p = pertubated points & how far they move it...
p = [1 for _ in range(len(lime_points))]
#iid = independent variables with identical distributions & put a 0 for each var in the range of length of X
iid = [0 for _ in range(len(X))]
#stacks to pertubations and the original standardized numpy X
all_x = np.vstack((lime_points,X))
#creates an array of perturbed points & points with the same probability distribution
#--> use for classifier; check to see if it's right
all_y = np.array(p + iid)

#downsizing the all_y pertubation data from 24688 to 6172
unbal_y = np.where((all_y == 0), 0, 1)
# print("Viewing the imbalanced target vector:\n", unbal_y)
class0_og = np.where(unbal_y == 0)[0]
class1_p = np.where(unbal_y == 1)[0]

class0_len = len(class0_og)
class1_len = len(class1_p)

print("class 0 length: ", class0_len)
print("class 1 length: ", class1_len)

class1_p_downsampled = np.random.choice(class1_p, size=class0_len, replace=False)
ds_all_y = np.hstack((unbal_y[class1_p_downsampled], unbal_y[class0_og]))

#downsample all_x using the indexes from ds_all_y
ds_xp = all_x[class1_p_downsampled]
ds_all_x = np.vstack((ds_xp, all_x[class0_og]))

print("-------------------------------------")
print("Data Downsized & Stacked Together")
print("-------------------------------------")

print("-------------------------------------")
print("PCA Begins")
print("-------------------------------------")
# Reduce the dimensionality of data to 2 features
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(ds_all_x)

print("PCA Results", pca_results)
print("Shape of PCA Results:", pca_results.shape[0])

# ------------------------------------------------------------------------------
# Recursive Divisive Clustering K = 2
# Adapted from: https://github.com/cwei01/Bisecting-K-means/blob/master/hw4_part2.ipynb
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
# ------------------------------------------------------------------------------

print("-------------------------------------")
print("Recursive Divisive Clustering K = 2")
print("-------------------------------------")
name=['og','perturbed']

#matplotlib inline
data = pd.DataFrame(pca_results)
data.columns = ['og','perturbed']
# data.head(5)
# plt.scatter(data.iloc[:,0],data.iloc[:,1])

from sklearn.cluster import KMeans
# import numpy as np
# import math
import warnings
warnings.filterwarnings("ignore")

#Calculate list to add elements multiple times
def fun1(list1,v,n):
    i=0
    while i<n:
        list1.append(v)
        i=i+1
    return list1
#Dictionary sorting to get tags
def fun2(list2):
    values = []
    y = []
    for i in range(len(list2)):
        values=fun1(values,i,len(list2[i]))
    keys=[n for a in list2 for n in a ]
    dic=dict(zip(keys,values))
    for i in sorted (dic) :
            y.append(dic[i])
    return y

def bisect(data, k, seed = 1):
    init_coor=[np.mean(data.iloc[:,0]),np.mean(data.iloc[:,1])]
    SSE=[]
    centroidlist = [data]
    sum=0
    for i in range(data.shape[0]):
        ds=(data.iloc[i,0]-init_coor[0])**2+(data.iloc[i,1]-init_coor[1])**2
        sum=sum+ds
    SSE.append(sum)
    print('Iteration 0  SSE =', SSE)
    datall=data
    datall_list=[]
    datall_list.append([i for i in range(0,240)])
    max_S_index=0
    for numClusters in range(1,k):
        SSE.pop(max_S_index)
        centroidlist.pop(max_S_index)
        datall_list.pop(max_S_index)
        clusters = KMeans(2, max_iter=1,random_state=seed)
        clusters.fit_predict(np.array(datall))
        lable_pred=clusters.labels_
        centroids=clusters.cluster_centers_
        centroidlist.append(centroids[0])
        centroidlist.append(centroids[1])
        SSE1=0
        SSE2=0
        dataA=pd.DataFrame(columns=['og','perturbed'])
        dataB=pd.DataFrame(columns=['og','perturbed'])
        for j in range(len(lable_pred)):
            if lable_pred[j]==False:
                d=(datall.iloc[j,0]-centroids[0][0])**2+(datall.iloc[j,1]-centroids[0][1])**2
                SSE1=SSE1+d
                dataA=dataA.append(datall.iloc[j])
            elif lable_pred[j]==True:
                e=(datall.iloc[j,0]-centroids[1][0])**2+(datall.iloc[j,1]-centroids[1][1])**2
                SSE2=SSE2+e
                dataB=dataB.append(datall.iloc[j])
        dataA_list= dataA.index.tolist()
        dataB_list= dataB.index.tolist()
        #print(dataB_list)
        #print(dataA_list)
        SSE.append(SSE1)
        SSE.append(SSE2)
        datall_list.append(dataA_list)
        datall_list.append(dataB_list)
        max_S_index=SSE.index(max(SSE))
        data_tmp=pd.DataFrame(columns=['og','perturbed'])
        data_tmp=data_tmp.append(data.iloc[[i for i in datall_list[max_S_index]]])
        datall=data_tmp
        print('Iteration', numClusters, ' SSE =', SSE)
    labels = fun2(datall_list)
    data['labels'] = labels
    return labels, centroidlist

data['labels'], centroidlist = bisect(data, int(math.sqrt(pca_results.shape[0])), 1)
plt.scatter(data['og'],data['perturbed'],c=data['labels'])
# Plot the centroids as a white X
cent = np.array(centroidlist)
print("CENTROID LENGTH:", len(cent))
# print("CENTROID ONE:", cent[1])
plt.scatter(
    cent[:,0],
    cent[:,1],
    marker=".",
    # s=169,
    linewidths=3,
    color="r",
    zorder=10,
)
plt.title(
    "Bisecting K-means clustering on the COMPAS dataset (PCA-reduced data)\n"
    "Centroids are marked with red"
)
plt.colorbar()
plt.show()

print("-------------------------------------")
print("Perturbation & OG Scatter SLACK Plot")
print("-------------------------------------")
#BLUE scatter plot( x = results[items from the beginning through 2000/stop-1 , column_0], y = results[items from the beginning through 2000/stop-1 , column_1], blending value 0-1)
# plt.scatter(pca_results[:2000,0], pca_results[:2000,1], alpha=.1)
#ORANGE scatter plot( x = results[idk but items start through the rest of the array , column_0], y = results[idk , column_1], blending value 0-1)
# plt.scatter(pca_results[-int(X.shape[0]*.10):,0], pca_results[-int(X.shape[0]*.10):,1], alpha=.1)
#plt.show()
