# Y(종속변수) 의 값이 양수의 값만 띄는 경우 회귀분석을 하게 되면 분석의 결과가 제대로 출력이 되지 않는다.
# 이 경우 Linear regression 에서 인자를 설정하여 변경해주어야 한다. 

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score 

np.random.seed(42)

n_samples, n_features = 200, 50 
x = np.random.randn(n_samples, n_features)

true_coef = 3 * np.random.randn(n_features)

# 모든 값을 양수로 처리해주기 위함.
true_coef[true_coef < 0] = 0 

y = np.dot(x, true_coef)

# add some noise 
y += 5 + np.random.normal(size = (n_samples, ))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

from sklearn.linear_model import LinearRegression

reg_nnls = LinearRegression(positive = True)
y_pred = reg_nnls.fit(x_train, y_train).predict(x_test)

r2_score_nnls = r2_score(y_test, y_pred)

print(f'NNLS R2 score : {r2_score_nnls}')


# fit an OLS
reg_ols = LinearRegression()
y_pred_ols = reg_ols.fit(x_train, y_train).predict(x_test)

r2_score_ols = r2_score(y_test, y_pred_ols)

print(f'OLS R2 Score {r2_score_ols}')


fig, ax = plt.subplots()
ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth = 0, marker = '.')
low_x, high_x = ax.get_xlim()
low_y, high_y = ax.get_ylim()

low = max(low_x, low_y)
high = min(high_x, high_y)

ax.plot([low, high], [low, high], ls = '--', c = '.3', alpha = 0.5)
ax.set_xlabel("OLS regression coefficient", fontweight = 'bold')
ax.set_ylabel("NNLS regression coefficient", fontweight = 'bold')
plt.show()