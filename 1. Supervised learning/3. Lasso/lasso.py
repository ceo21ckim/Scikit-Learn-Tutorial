from sklearn.linear_model import Lasso, LassoCV 
reg = Lasso(alpha = 0.1)

x_data = [[0, 0], [1, 1]]
y_data = [0, 1]

reg.fit(x_data, y_data)

x_test = [[1, 1]]

reg.predict(x_test)
# 0.8

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score 

np.random.seed(42)

n_samples, n_features = 50, 100 
x = np.random.randn(n_samples, n_features)

idx = np.arange(n_features)
coef = (-1) ** idx * np.exp(-idx / 10 )

coef[10:] = 0 
y = np.dot(x, coef)

# bias, noise
y += 0.01 * np.random.normal(size = n_samples )

n_samples = x.shape[0]
x_train, y_train = x[: n_samples // 2], y[: n_samples // 2]
x_test, y_test = x[n_samples//2 : ], y[n_samples //2 : ]

# Lasso regression ----
from sklearn.linear_model import Lasso 
alpha = 0.1
lasso = Lasso(alpha = alpha )

y_pred_lasso = lasso.fit(x_train, y_train).predict(x_test)

r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print(f"r^2 on test data : {r2_score_lasso:.4f}")

m, s, _ = plt.stem(
    np.where(lasso.coef_)[0], 
    lasso.coef_[lasso.coef_ != 0], 
    markerfmt = 'x', 
    label = "Lasso coefficients", 
    use_line_collection=True,
)
plt.setp([m, s], color = '#ff7f0e')
plt.stem(
    np.where(coef)[0], 
    coef[coef != 0], 
    label = 'true coefficients', 
    markerfmt = 'bx', 
    use_line_collection=True,
)
plt.legend(loc = 'best')
plt.title(
    f'Lasso $R^2$ : {r2_score_lasso:.3f}'
)
plt.show()