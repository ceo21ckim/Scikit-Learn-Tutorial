import numpy as np
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt 


np.random.seed(42)

n_samples, n_features = 50, 100
X = np.random.randn(n_samples, n_features)

# Decreasing coef w. alternated signs for visualization
idx = np.arange(n_features)
coef = (-1) ** idx * np.exp(-idx / 10)
coef[10:] = 0  # sparsify coef
y = np.dot(X, coef)

# Add noise
y += 0.01 * np.random.normal(size=n_samples)

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[: n_samples // 2], y[: n_samples // 2]
X_test, y_test = X[n_samples // 2 :], y[n_samples // 2 :]

from sklearn.linear_model import ElasticNet
alpha = 0.1
enet = ElasticNet(alpha = alpha, l1_ratio = 0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)

print(enet)
print(f' r^2 on the data {r2_score_enet:.4f}')

m, s, _ = plt.stem(
    np.where(enet.coef_)[0], 
    enet.coef_[enet.coef_ != 0], 
    markerfmt = 'x', 
    label = 'Elastic net coefficients', 
    use_line_collection = True,
)
plt.step([m, s], color = '#2ca02c')

plt.stem(
    np.where(coef)[0],
    coef[coef != 0],
    label="true coefficients",
    markerfmt="bx",
    use_line_collection=True,
)

plt.legend(loc = 'best')
plt.title(
    f"Elastic Net $R^2$: {r2_score_enet}"
)
plt.show()