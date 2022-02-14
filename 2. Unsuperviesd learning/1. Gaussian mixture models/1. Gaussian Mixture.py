# GMM covariances

import matplotlib as mpl
from matplotlib import transforms 
import matplotlib.pyplot as plt

import numpy as np 
from sklearn import datasets
from sklearn.mixture import GaussianMixture 
from sklearn.model_selection import train_test_split, StratifiedKFold

colors = ['navy', 'turquoise', 'darkorange']

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariance_[n][:2, :2]
        
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
            
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            
        v, w = np.linalg.eigh(covariances) # output : eigen vector, eigen value
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi 
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], 180 + angle, color = color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')
        
iris = datasets.load_iris()

skf = StratifiedKFold(n_splits = 4)

train_index, test_index = next(iter(skf.split(iris.data, iris.target)))

x_train = iris.data[train_index]
y_train = iris.target[train_index]

x_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = np.unique(y_train).__len__()

estimators = {
    cov_type : GaussianMixture(
        n_components = n_classes, covariance_type = cov_type, max_iter = 20, random_state = 0 
    )
    for cov_type in ['spherical', 'diag', 'tied', 'full']
}

n_estimators = len(estimators)

plt.figure(figsize = (3 * n_estimators // 2, 6))
plt.subplots_adjust(
    bottom = 0.01, top = 0.95, hspace = 0.15, wspace = 0.05, left = 0.01, right = 0.09
)

for index, (name, estimator) in enumerate(estimators.items()):
    estimator.means_init = np.array(
        [x_train[y_train == i].mean(axis = 0) for i in range(n_classes)]
    )
    
    estimator.fit(x_train)
    
    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)
    
    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(
            data[:, 0], data[:, 1], s = 0.8, color = color, label = iris.target_names[n]
        )
        
    for n, color in enumerate(colors):
        data = x_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker = 'x', color = color)
        
    y_train_pred = estimator.predict(x_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100 
    plt.text(0.05, 0.9, f'Train accuracy: {train_accuracy:.1f}', transform = h.transAxes)
    
    y_test_pred = estimator.predict(x_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100 
    plt.text(0.05, 0.8, f"Test accuracy: {test_accuracy:.1f}", transform = h.transAxes)
    
    plt.xticks(())
    plt.yticks(())
    plt.title(name)
    
plt.legend(scatterpoints = 1, loc = 'lower right', prop = dict(size = 12))
plt.show()