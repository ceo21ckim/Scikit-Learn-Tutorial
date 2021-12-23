# MNIST classification using multinomial

import time 
import matplotlib.pyplot as plt 
import numpy as np 


from sklearn.datasets import fetch_openml 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.utils import check_random_state

x, y = fetch_openml('mnist_784', version = 1, return_X_y=True, as_frame = False)
t0 = time.time()
train_samples = 50000

random_state = check_random_state(0)
permutation = random_state.permutation(x.shape[0])

x = x[permutation]
y = y[permutation]
x = x.reshape((x.shape[0], -1))


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = train_samples, test_size = 10000
)

# scaler
scaler = StandardScaler()
x_train  = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

clf_default = LogisticRegression(solver = 'saga')
clf_tune = LogisticRegression(C = 50.0 / train_samples, penalty = 'l1', solver = 'saga', tol = 0.1)

clf_default.fit(x_train, y_train)
clf_tune.fit(x_train, y_train)

# penalty : l1 / l1 정규화를 사용했기 때문에 변수들이 제거되는데 얼마나 제거되었는가 보기 위함.
sparsity_default = np.mean(clf_default.coef_ == 0) * 100
sparsity_tune = np.mean(clf_tune.coef_ == 0) * 100

score_default = clf_default.score(x_test, y_test)
score_tune = clf_tune.score(x_test, y_test)

print(f"Sparsity with L1 penalty (default) : {sparsity_default:.2f}")
print(f"Sparsity with L1 penalty (tune) : {sparsity_tune:.2f}")

print(f'Test score with L1 penalty (default) : {score_default:.4f}')
print(f'Test score with L1 penalty (tune) : {score_tune:.4f}')

coef_default = clf_default.coef_.copy()
coef_tune = clf_tune.coef_.copy()
scale_default = np.abs(coef_default).max()
scale_tune = np.abs(coef_tune).max()

num_classes = np.unique(y).shape[0]

for i in range(num_classes):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(
        coef_default[i].reshape(28, 28), 
        interpolation = 'nearest', 
        cmap = plt.cm.RdBu, 
        vmin = -scale_default, 
        vmax = scale_default,
    )
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel("Class %i" %i)
plt.suptitle("Classification vector for...")
plt.show()

run_time = time.time() - t0 
print('Example run in %.3f s' % run_time)

for i in range(num_classes):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(
        coef_tune[i].reshape(28, 28), 
        interpolation = 'nearest', 
        cmap = plt.cm.RdBu, 
        vmin = -scale_tune, 
        vmax = scale_tune,
    )
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel("Class %i" %i)
plt.suptitle("Classification vector for...")

run_time = time.time() - t0 
print('Example run in %.3f s' % run_time)
plt.show()
