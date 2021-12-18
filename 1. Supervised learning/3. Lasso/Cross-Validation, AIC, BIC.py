import time
from matplotlib.lines import _LineStyle 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.linear_model import (
    LassoCV, 
    LassoLarsCV, 
    LassoLarsIC
)

from sklearn import datasets 

# This is to avoid division by zero while doing np.log10
epsilon = 1e-4 

x, y = datasets.load_diabetes(return_X_y=True)

rng = np.random.RandomState(42)

# np.c_ == np.concatenate(axis = 1)
# x.shape : (442, 10) -> (442, 14)
x = np.c_[x, rng.randn(x.shape[0], 14)] # add some bad features

# normalize data as don by Lars to allow for comparison
x /= np.sqrt(np.sum(x **2, axis = 0))

# LassoLarsIC : least anle regression with BIC/AIC criterion
model_bic = LassoLarsIC(criterion = 'bic', normalize = False)
t1 = time.time()
model_bic.fit(x, y)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_ 

model_aic = LassoLarsIC(criterion = 'aic', normalize = False)
model_aic.fit(x, y)
alpha_aic_ = model_aic.alpha_



def plot_ic_criterion(model, name, color):
    criterion_ = model.criterion_ 
    plt.semilogx(
        model.alphas_ + epsilon, 
        criterion_, 
        '--', 
        color = color, 
        linewidth = 3, 
        label = f'{name:s} criterion'
    )
    plt.axvline(
        model.alpha_ + epsilon, 
        color = color, 
        linewidth = 3, 
        label = f'alpha : {name:s} estimate',    
    )
    plt.xlabel(r'$\alpha$')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, "AIC", 'b')
plot_ic_criterion(model_bic, "BIC", 'r')
plt.legend()
plt.title("Information-criterion for model selection (training time %.3fs" % t_bic)

####################################################################################
# LassoCV: coordinate descent

print("Computing regularization path using the coordinate descent lasso...")
t1 = time.time()
model = LassoCV(cv = 20).fit(x, y)
t_lasso_cv = time.time() - t1

plt.figure()
ymin, ymax = 2300, 2300
plt.semilogx(model.alpha_ + epsilon, model.mse_path_, ':')
plt.plot(
    model.alphas_ + epsilon, 
    model.mse_path_.mean(axis = -1), 
    'k', 
    label = 'Average across the folds', 
    linewidth = 2, 
)
plt.axvline(
    model.alpha_ + epsilon, linestyle = '--', color = 'k', label = 'alpha: CV estimate'
)
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel('Mean square error')
plt.title(
    "Mean sqaure error on each fold: coordinate descent (train time : %.2fs)" % t_lasso_cv
)
plt.axis('tight')
plt.ylim(ymin, ymax)

####################################################################################
# LassoLarsCV: least angle regression 

print("Computing regularization path using the Lars lasso...")
t1 = time.time()
model = LassoLarsCV(cv = 20, normalize = False).fit(x, y)
t_lasso_lars_cv = time.time() - t1

plt.figure()
plt.semilogx(model.cv_alphas_ + epsilon, model.mse_path_, ':')
plt.semilogx(
    model.cv_alphas_ + epsilon, 
    model.mse_path_.mean(axis = -1), 
    'k',
    label = 'Average across the folds', 
    linewidth = 2,
)
plt.axvline(model.alpha_, linestyle = '--', color = 'k', label = 'alpha CV')
plt.legend()

plt.xlabel(r'$\alpha$')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)' % t_lasso_lars_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)

plt.show()