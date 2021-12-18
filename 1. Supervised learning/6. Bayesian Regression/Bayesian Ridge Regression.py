# Bayesian Ridge Regression ----
from sklearn.linear_model import BayesianRidge
import numpy as np 
x = np.array([[0, 0], [1, 1], [2, 2], [3, 3]]).astype(float)
y = np.array([0, 1, 2, 3]).astype(float)

reg = BayesianRidge()
reg.fit(x, y)

x_test = np.array([[1., 0.]])
reg.predict(x_test)

reg.coef_

####################################################################

import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 

from sklearn.linear_model import (
    BayesianRidge, 
    LinearRegression
)

np.random.seed(0)
n_samples, n_features = 100, 100
x = np.random.randn(n_samples, n_features) # Create Gaussian data 

lambda_ = 4.0
w = np.zeros(n_features)

relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(lambda_))

alpha_ = 50.0 
noise = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(alpha_), size=n_samples)

y = np.dot(x, w) + noise 

# fit the bayesian ridge regression and ols for comparison
clf = BayesianRidge(compute_score = True)
clf.fit(x, y)

ols = LinearRegression()
ols.fit(x, y)

lw = 2
plt.figure(figsize = (6, 5))
plt.title("Weights of the model")
plt.plot(clf.coef_, color = 'lightgreen', linewidth = lw, label = 'Bayesian Ridge estimate')
plt.plot(w, color = 'gold', linewidth = lw, label = 'Ground truth')
plt.plot(ols.coef_, color = 'navy', linestyle = '--', label = 'OLS estimate')
plt.xlabel('Features')
plt.ylabel('Values of the weights')
plt.legend(loc = 'best', prop=dict(size = 12))

plt.figure(figsize = (6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins = n_features, color = 'gold', log = True, edgecolor = 'black')
plt.scatter(
    clf.coef_[relevant_features], 
    np.full(len(relevant_features), 5.0), 
    color = 'navy', 
    label = 'Relevant features'
)
plt.ylabel("Features")
plt.xlabel('Values of the weights')
plt.legend(loc = 'upper left')

plt.figure(figsize = (6, 5))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_, color = 'navy', linewidth = lw)
plt.ylabel("score")
plt.xlabel('Iterations')

def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise 

degree = 10
x = np.linspace(0, 10, 100)
y = f(x, noise_amount = 0.1)
clf_poly = BayesianRidge()
clf_poly.fit(np.vander(x, degree), y)

x_plot = np.linspace(0, 11, 25)
y_plot = f(x_plot, noise_amount = 0 )
y_mean, y_std = clf_poly.predict(np.vander(x_plot, degree), return_std = True)
plt.figure(figsize = (6, 5))
plt.errorbar(
    x_plot, 
    y_mean, 
    y_std, 
    color = 'navy', 
    label = 'Polynomial Bayesian Ridge Regression', 
    linewidth = lw,
)

plt.plot(x_plot, y_plot, color = 'gold', linewidth = lw, label = "Ground Truth")
plt.ylabel("Output y")
plt.xlabel("Feature x")
plt.legend(loc = 'lower left')
plt.show()