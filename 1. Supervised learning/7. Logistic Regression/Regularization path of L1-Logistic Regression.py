from time import time 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn import (
    linear_model, 
    datasets
)

from sklearn.svm import l1_min_c 

iris = datasets.load_iris()
x = iris.data 
y = iris.target 

x = x[y != 2]
y = y[y != 2]

x /= x.max() 

cs = l1_min_c(x, y, loss = 'log') * np.logspace(0, 7, 16)

print("Computing regularization path ...")
start = time()
clf = linear_model.LogisticRegression(
    penalty = 'l1', 
    solver = 'liblinear', 
    tol = 1e-6, 
    max_iter = int(1e6), 
    warm_start = True, 
    intercept_scaling=10000.0, 
)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(x, y)
    coefs_.append(clf.coef_.ravel().copy())
    
print(f"This took {time()-start:.3f}")

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_, marker = 'o')
ymin, ymax = plt.ylim()
plt.xlabel("log(C)")
plt.ylabel("Coefficients")
plt.title("Logistic Regression Path")
plt.axis('tight')
plt.show()