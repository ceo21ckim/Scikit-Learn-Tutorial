from sklearn import linear_model
import numpy as np
# linear Regression 

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
x_data = [[0, 0], [1, 1], [2, 2]]
y_data = [0, 1, 2]

reg.fit(x_data, y_data)

reg.coef_
# array([0.5, 0.5])

x_data_array = np.array(x_data)
coef = reg.coef_

coef*x_data_array

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

x_data, y_data = datasets.load_diabetes(return_X_y=True)

# 새로운 차원을 만들어주는 함수.
np.newaxis

x_data[:, 2].shape
new_x = x_data[:, 2, np.newaxis]

# split train, test
x_train = new_x[:-20]
x_test = new_x[-20:]

y_train = y_data[:-20]
y_test = y_data[-20:]
regression = LinearRegression()

regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

# coefficient 
print(f"Coefficient : {regression.coef_}")

# mean squared error
print(f'mean squared error : {mean_squared_error(y_test, y_pred):.3f}')

# r square
print(f'Coefficient of determination : {r2_score(y_test, y_pred):.3f}')

plt.scatter(x_test, y_test, color = 'black')
plt.plot(x_test, y_pred, color = 'blue', linewidth = 3)
plt.xticks(())
plt.yticks(())
plt.show()