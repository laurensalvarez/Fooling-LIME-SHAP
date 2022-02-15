# Adapted from: Kernel Support Vector Machine https://github.com/sametgirgin/Machine-Learning-Classification-Models/blob/master/Kernel%20Support%20Vector%20Machine.py
from adversarial_models import *
from utils import *
from get_data import *

#1 Importing the libraries
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import pandas as pd

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
print("Viewing the imbalanced target vector:\n", unbal_y)
class0_og = np.where(unbal_y == 0)[0]
class1_p = np.where(unbal_y == 1)[0]

class0_len = len(class0_og)
class1_len = len(class1_p)

print("class 0 length: ", class0_len)
print("class 1 length: ", class1_len)

class1_p_downsampled = np.random.choice(class1_p, size=class0_len, replace=False)
ds_all_y = np.hstack((unbal_y[class1_p_downsampled], unbal_y[class0_og]))
print("class1_p_downsampled.shape")
print(class1_p_downsampled.shape)
print("class1_p_downsampled")
print(class1_p_downsampled[0:5])
print("class 0 length: ", class0_len)
print("class1_p_downsampledlength: ", str(len(class1_p_downsampled)))
print(); print(np.hstack((unbal_y[class1_p_downsampled], unbal_y[class0_og])))
print("ds_all_y.shape")
print(ds_all_y.shape)

#downsample all_x using the indexes from ds_all_y
ds_xp = all_x[class1_p_downsampled]
ds_all_x = np.vstack((ds_xp, all_x[class0_og]))

print("ds_all_x.shape")
print(ds_all_x.shape)


#4 Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ds_all_x, ds_all_y, test_size = 0.20, random_state = 0, shuffle= True)

#4 Feature Scaling
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)

#5 Fitting Kernel SVM classifier to the Training set
# Create your classifier object here
# from sklearn.svm import SVC, OneClassSVM
# from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score
#
# # k_fold = KFold(n_splits = 10,random_state=None, shuffle=True)
# rk_fold = RepeatedKFold(n_splits = 5, n_repeats=5, random_state=None)
# #Lets choose Gaussian kernel
# classifier = SVC(kernel='rbf', random_state=0)
# [classifier.fit(X_train, y_train) for train, test in rk_fold.split(X_train)]
# scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=rk_fold)
# #report performance
# print ('Cross_val Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# #classifier.fit(X_train, y_train)
#
# #6 Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# #7 Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# cm = confusion_matrix(y_test, y_pred)
# print ("confusion matrix of y_test & y_pred [[TP, FP; FN, TN]]: " , cm)
# print(classification_report(y_test, y_pred, target_names= ["Original" , "Perturbed"]))

print("-------------------------------------")
print("OOD/Outlier detection")
print("-------------------------------------")
#OOD/Outlier detection
# Isolation Forest is based on the Decision Tree algorithm.
# It isolates the outliers by randomly selecting a feature from the given set of
# features and then randomly selecting a split value between the max and min values of that feature.
# This random partitioning of features will produce shorter paths in trees for
# the anomalous data points, thus distinguishing them from the rest of the data.
# Slack used a RF
from sklearn.ensemble import IsolationForest
from numpy import where
# fit the model
rng = np.random.RandomState(42)
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=20, edgecolor="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="green", s=20, edgecolor="k")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="red", s=20, edgecolor="k")
plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend(
    [b1, b2, c],
    ["training observations", "new regular observations", "new abnormal observations"],
    loc="upper left",
)
plt.show()

# #using pca to reduce dimensionality
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# Xreduced = pca.fit_transform(X_test)



#8 Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# # plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
# #              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(7)]).T
# pred = classifier.decision_function(Xpred).reshape(X1.shape)
# plt.contourf(X1, X2, pred,
#              alpha=1.0, cmap="RdYlGn", levels=np.linspace(pred.min(), pred.max(), 100))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Kernel SVM Classifier (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
#
# #9 Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Kernel SVM Classifier (Test set)')
# plt.xlabel('Default: Age/to')
# plt.ylabel('Default: Estimated Salary')
# plt.legend()
# plt.show()
