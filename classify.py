# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:21:25 2022

@author: yin
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
#load data
data = pd.read_csv('data.txt', sep='\t', names=['sp', 'complexity', 'label', 'p1', 'p2'])
X = data[['sp', 'complexity']].values
y = data['label'].values
#check

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
print(len(y) / sum(y), len(y_train) / sum(y_train), len(y_test) / sum(y_test))
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import SelfTrainingClassifier

i=1  
print('random seed ', i)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
# 分布比例
print(len(y) / sum(y), len(y_train) / sum(y_train), len(y_test) / sum(y_test))
rng = np.random.RandomState(2 * i)
y_rand = rng.rand(y_train.shape[0])

y_5 = np.copy(y_train)
y_5[y_rand > 0.05] = -1  # set random samples to be unlabeled
y_10 = np.copy(y_train)
y_10[y_rand > 0.1] = -1
y_20 = np.copy(y_train)
y_20[y_rand > 0.2] = -1
print(sum(y_5 == 0) / sum(y_5 == 1), sum(y_10 == 0) / sum(y_10 == 1), sum(y_20 == 0) / sum(y_20 == 1))

ls5 = (LabelSpreading().fit(X_train, y_5), y_5, "Label Spreading 5% data")
ls10 = (LabelSpreading().fit(X_train, y_10), y_10, "Label Spreading 10% data")
ls20 = (LabelSpreading().fit(X_train, y_20), y_20, "Label Spreading 20% data")
ls100 = (LabelSpreading().fit(X_train, y_train), y_train, "Label Spreading 100% data")

print(ls5[0].score(X_test, y_test), ls5[2])
print(ls10[0].score(X_test, y_test), ls10[2])
print(ls20[0].score(X_test, y_test), ls20[2])
print(ls100[0].score(X_test, y_test), ls100[2])

Ls=LabelSpreading()
Ls.fit(X_train, y_5)
pre_y_5 = Ls.predict(X_test)

Ls=LabelSpreading()
Ls.fit(X_train, y_10)
pre_y_10 = Ls.predict(X_test)

Ls=LabelSpreading()
Ls.fit(X_train, y_20)
pre_y_20 = Ls.predict(X_test)

base_classifier = SVC(kernel="linear", probability=True)
st5 = (SelfTrainingClassifier(base_classifier).fit(X_train, y_5), y_5, "Self Training 5% data")
st10 = (SelfTrainingClassifier(base_classifier).fit(X_train, y_10), y_10, "Self Training 10% data")
st20 = (SelfTrainingClassifier(base_classifier).fit(X_train, y_20), y_20, "Self Training 20% data")
st100 = (SelfTrainingClassifier(base_classifier).fit(X_train, y_train), y_train, "Self Training 100% data")

print(st5[0].score(X_test, y_test), st5[2])
print(st10[0].score(X_test, y_test), st10[2])
print(st20[0].score(X_test, y_test), st20[2])
print(st100[0].score(X_test, y_test), st100[2])

St=SelfTrainingClassifier(base_classifier)
St.fit(X_train, y_5)
pre_y_5_st = St.predict(X_test)

St=SelfTrainingClassifier(base_classifier)
St.fit(X_train, y_10)
pre_y_10_st = St.predict(X_test)

St=SelfTrainingClassifier(base_classifier)
St.fit(X_train, y_20)
pre_y_20_st = St.predict(X_test)


St=SelfTrainingClassifier(base_classifier)
St.fit(X_train, y_train)
pre_y_100_st = St.predict(X_test)

x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
h=0.09
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
color_map = {0: (0, 0, 0.9), 1: (0.8,0.6,0)}

classifiers = (ls5,st5,ls10,st10,ls20,st20,ls100,st100)
for i, (clf,y_train,title) in enumerate(classifiers):
    plt.subplot(4,2,i+1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    Z_pre=clf.predict(X_test)
    plt.axis("off")
    y_wrong=Z_pre == y_test
    colors = [color_map[y] for y in Z_pre]
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, edgecolors="black")
    for j in range(len(y_wrong)):
        if y_wrong[j] == False:
            plt.scatter(X_test[j,0], X_test[j,1], color='none', marker='o', edgecolors='black', s=200)
    plt.title(title)
    
plt.savefig('pic.pdf', dpi=300)
plt.show()
#SVM

clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, X, y, cv=3)
clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, X, y, cv=2)
clf = svm.SVC(kernel='linear', class_weight='balanced')
scores = cross_val_score(clf, X, y, cv=3)
clf = svm.SVC(kernel='rbf')
scores = cross_val_score(clf, X, y, cv=3)
clf = svm.SVC(kernel='rbf', class_weight='balanced')
scores = cross_val_score(clf, X, y, cv=3)
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
clf = LDA(solver="svd", store_covariance=True)
scores = cross_val_score(clf, X, y, cv=3)
clf = LDA(solver="svd", store_covariance=True)
scores = cross_val_score(clf, X, y, cv=2)
clf = QDA(store_covariance=True)
scores = cross_val_score(clf, X, y, cv=3)
clf = QDA(store_covariance=True)
scores = cross_val_score(clf, X, y, cv=2)
#kmeans
from sklearn.cluster import KMeans
size = len(y)
y_pred = KMeans(n_clusters=2).fit_predict(X)
sum(y_pred == y) / size, sum(1 - y_pred == y) / size
plt.scatter(X[:, 0], X[:, 1], c=y, s=2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=2)
plt.scatter(X[:, 0], X[:, 1], c=1 - y_pred, s=2)
#GMM
from sklearn import mixture
dpgmm = mixture.BayesianGaussianMixture(n_components=4, 
                                        covariance_type="full", 
                                        weight_concentration_prior_type='dirichlet_distribution',
                                        random_state=2333,
                                       ).fit(X)
y_pred = dpgmm.predict(X)
y_pred
print(sum(y_pred == y) / size, sum(1 - y_pred == y) / size)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=2)
#normalize data

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = LDA(solver="svd", store_covariance=True)
scores = cross_val_score(clf, X_scaled, y, cv=3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, s=2)
clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, X_scaled, y, cv=3)
clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, X_scaled, y, cv=2)
clf = svm.SVC(kernel='rbf')
scores = cross_val_score(clf, X_scaled, y, cv=3)
