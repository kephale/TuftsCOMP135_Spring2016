
# Expectation Maximization Lecture

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

np.random.seed( 2507365 ) # We'll set the random number generator's seed so everyone generates the exact same dataset

# Plot a Gaussian

import math
import matplotlib.mlab as mlab
mean = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(-3,3,100)
plt.plot(x,mlab.normpdf(x,mean,sigma))
plt.savefig('../images/example_gaussian_1D.svg')

# Flipping coins

thetaA = 0.4
thetaB = 0.6

z = []
heads = []
num_flips = 10

for i in range(5):
    if np.random.rand() > 0.5:
        theta = thetaA
        z += ['a']
    else:
        theta = thetaB
        z += ['b']


    heads += [sum( np.random.rand(num_flips) < theta )]

# Scikit Learn GMM + EM example:

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

# Show K-Means performance
def run_with_k( k ):
    est = KMeans(n_clusters=k)

    plt.cla()
    est.fit( np.array(X) )
    labels = est.labels_

    color_map = [ np.random.rand(3) for _ in range( k ) ]
    colors = [ color_map[label] for label in labels ]

    #plt.scatter(points_x, points_y, c=colors)

    return est
    #return labels, est.cluster_centers_
    #return cluster_scatter( points, labels )
    #return cluster_distance( points, labels )
    #return cluster_summin_distance( points, labels )

color_map = [ np.random.rand(3) for _ in range( k ) ]

# Find clusters with k=2
k = 2

#kmean_labels,kmean_centers = run_with_k(k)
est = run_with_k(k)
kmean_labels = est.labels_
kmean_centers = est.cluster_centers_
dists = est.transform( np.array(X) )
mx0 = np.max( dists[:,0] )
mx1 = np.max( dists[:,1] )
dists = [ np.array([r[0]/mx0,0,r[1]/mx1,1]) for r in dists ]

ax = plt.axes()

points_x = [ el[0] for el in X ]
points_y = [ el[1] for el in X ]

colors = [ color_map[label] for label in kmean_labels ]

# Draw cluster labels

plt.scatter(points_x, points_y, c=colors)
plt.scatter(kmean_centers[:,0], kmean_centers[:,1], c=[ color_map[label] for label in range(k) ], s=100)

plt.savefig('../images/kmeans_k=' + str(k) + '_incorrect.svg')

# Draw cluster distances

#plt.gca()
plt.scatter(points_x, points_y, c=dists,s=75)
#plt.scatter(kmean_centers[:,0], kmean_centers[:,1], c=[ color_map[label] for label in range(k) ], s=100)

plt.savefig('../images/kmeans_k=' + str(k) + '_incorrect_dists.svg')


#plt.plot( X, Y )

#ax.set_xlabel('k')
#ax.set_ylabel('Cluster scatter')

# Show different # latent variables for EM
lowest_bic = np.infty
bic = []
#n_components_range = range(1, 7)
n_components_range = range(2, 3)
cv_types = ['spherical', 'tied', 'diag', 'full']
#cv_types = ['diag']
cv_types = ['full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type )
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
clf = best_gmm
bars = []

# Plot the BIC scores
#spl = plt.subplot(2, 1, 1)
# for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#     xpos = np.array(n_components_range) + .2 * (i - 2)
#     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#                                   (i + 1) * len(n_components_range)],
#                         width=.2, color=color))
# plt.xticks(n_components_range)
# plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
# plt.title('BIC score per model')
# xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(bic.argmin() / len(n_components_range))
# plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
# spl.set_xlabel('Number of components')
# spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
#splot = plt.subplot(2, 1, 2)
splot = plt.subplot(111)
Y_ = clf.predict(X)
for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_,
                                             color_iter)):

    v, w = linalg.eigh(covar)

    # for diag or spherical
    #v, w = linalg.eigh(np.diagflat(covar))
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180 * angle / np.pi  # convert to degrees
    v *= 4
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)

plt.xlim(-10, 10)
plt.ylim(-3, 6)
plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model, 2 components')
plt.subplots_adjust(hspace=.35, bottom=.02)
#plt.savefig('../images/em_k=' + str(k) + '_cov_spherical.svg')
plt.savefig('../images/em_k=' + str(k) + '_cov_full.svg')
#plt.show()
#plt.savefig('../images/em_k=' + str(k) + '_cov_spherical.png')



# Walk through EM step by step:

lowest_bic = np.infty
bic = []
#n_components_range = range(1, 7)
n_components_range = range(2, 3)
cv_types = ['spherical', 'tied', 'diag', 'full']
#cv_types = ['diag']
cv_types = ['full']
for iteration in range(1, 8):
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, verbose=2, n_iter=iteration)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    # Plotting
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    plt.cla()
    splot = plt.subplot(111)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_,
                                                 color_iter)):

        v, w = linalg.eigh(covar)

        # for diag or spherical
        #v, w = linalg.eigh(np.diagflat(covar))
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 4
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xlim(-10, 10)
    plt.ylim(-3, 6)
    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    #plt.savefig('../images/em_k=' + str(k) + '_cov_spherical.svg')
    plt.savefig('../images/em_k=' + str(k) + '_cov_full_step=' + str(iteration) + '.svg')



bic = np.array(bic)
color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
clf = best_gmm
bars = []
