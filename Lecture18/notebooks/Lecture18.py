%matplotlib inline

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.spatial
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn import mixture
from scipy.io import arff
from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import label_propagation
from sklearn import preprocessing

import os
import os.path
import random
import pickle

rng = np.random.RandomState(0)

image_directory = '/Users/kyle/git/TuftsCOMP135_Spring2016/Lecture18/images/'

# https://archive.ics.uci.edu/ml/datasets/Iris
iris = datasets.load_iris()

fullX = iris.data[:, :2]
fullX1 = iris.data[:, 0]
fullX2 = iris.data[:, 1]
fully = iris.target

X = []
X1 = []
X2 = []
y = []
for k in range(len(fully)):
    if fully[k] != 2:
        X += [ fullX[k] ]
        y += [ fully[k] ]
        X1 += [ fullX1[k] ]
        X2 += [ fullX2[k] ]

colors = []
color_map = [ (1,0,0), (0,0,1) ]
for c in y:
    if c == 0:
        colors += [ color_map[0] ]
    else:
        colors += [ color_map[1] ]
areas = [ 80 for _ in range(len(X)) ]
plt.scatter( X1, X2, c=colors, s=areas )
ax=plt.gca()
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
handles, labels = ax.get_legend_handles_labels()
#plt.show()
plt.savefig(image_directory+'iris_setosa_versicolor.png')

# SVM: Plot the margin


nX = preprocessing.scale(X)
nX1 = [ r[0] for r in nX ]
nX2 = [ r[1] for r in nX ]

clf = svm.SVC(kernel='linear')
clf.fit(nX, y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=150, facecolors='none')
#plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.scatter( nX1, nX2, c=colors, s=areas )
ax=plt.gca()
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')

plt.axis('tight')
plt.show()

# Plotting kernels for Iris

# Our dataset and targets
nX = preprocessing.scale(X)
nX1 = [ r[0] for r in nX ]
nX2 = [ r[1] for r in nX ]

# figure number
fignum = 1

# fit the model
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(nX, y)

    # plot the line, the points, and the nearest vectors to the plane
    #plt.figure(fignum, figsize=(4, 3))
    plt.figure(fignum)
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(nX1, nX2, c=y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    #plt.figure(fignum, figsize=(4, 3))
    plt.figure(fignum)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
    plt.savefig(image_directory+'svm_kernelexample_' + kernel + '_iris.svg')

# Noisy version of the data
rng = np.random.RandomState(0)
noise_level = 0.9
noisyX1 = [ x + noise_level * np.random.rand() for x in X1 ]
noisyX2 = [ x + noise_level * np.random.rand() for x in X2 ]
areas = [ 80 for _ in range(len(X)) ]
plt.scatter( noisyX1, noisyX2, c=colors, s=areas )
ax=plt.gca()
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
handles, labels = ax.get_legend_handles_labels()
#plt.show()
plt.savefig(image_directory+'iris_setosa_versicolor_noisy.svg')

# SVM: Plot the margin


nX = preprocessing.scale( np.array( zip( noisyX1, noisyX2 )) )
nX1 = [ r[0] for r in nX ]
nX2 = [ r[1] for r in nX ]

clf = svm.SVC(kernel='linear')
clf.fit(nX, y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=150, facecolors='none')
#plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.scatter( nX1, nX2, c=colors, s=areas )
ax=plt.gca()
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')

plt.axis('tight')
#plt.show()
plt.savefig(image_directory+'iris_setosa_versicolor_noisy_svm.svg')

# Scikit version

# we create 40 separable points
np.random.seed(0)
skX = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
skY = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(skX, skY)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k--')
plt.plot(xx, yy_down, 'k-')
plt.plot(xx, yy_up, 'k-')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(skX[:, 0], skX[:, 1], c=skY, cmap=plt.cm.Paired)

plt.axis('tight')
#plt.show()
#plt.savefig(image_directory+'random_data_margins.svg')
#plt.savefig(image_directory+'svm_decisionrule.png')
#plt.savefig(image_directory+'svm_constraints.png')
plt.savefig(image_directory+'svm_vector_within_margin.png')

# Scikit Learn's SVM kernels

# Our dataset and targets
skX = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
skY = [0] * 8 + [1] * 8

# figure number
fignum = 1

# fit the model
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(skX, skY)

    # plot the line, the points, and the nearest vectors to the plane
    #plt.figure(fignum, figsize=(4, 3))
    plt.figure(fignum)
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(skX[:, 0], skX[:, 1], c=skY, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    #plt.figure(fignum, figsize=(4, 3))
    plt.figure(fignum)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
    plt.savefig(image_directory+'svm_kernelexample_' + kernel + '_sk.svg')
#plt.show()

### Draw multiple steps of SVM

num_steps = 20

for steps in range( 1, num_steps ):

    # fit the model
    clf = svm.SVC(kernel='linear',max_iter=steps)
    clf.fit(skX, skY)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k--')
    plt.plot(xx, yy_down, 'k-')
    plt.plot(xx, yy_up, 'k-')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=80, facecolors='none')
    plt.scatter(skX[:, 0], skX[:, 1], c=skY, cmap=plt.cm.Paired)

    plt.axis('tight')
    #plt.show()
    #plt.savefig(image_directory+'random_data_margins.svg')
    #plt.savefig(image_directory+'svm_decisionrule.png')
    #plt.savefig(image_directory+'svm_constraints.png')
    #plt.savefig(image_directory+'svm_vector_within_margin.png')

    plt.savefig(image_directory+'random_data_svm_step_' + str(steps) + '.svg')
