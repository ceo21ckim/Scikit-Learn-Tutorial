import numpy as np
from numpy.lib.function_base import median 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.datasets import fetch_openml 

# if as_frame -> load Data.frame 
survey = fetch_openml(data_id = 534, as_frame = True)

survey.target

x = survey.data[survey.feature_names]
# 다차원 배열을 1차원 배열로 평평하게 펴주는 ravel, flatten, squeeze
y = survey.target.values.ravel()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

train_dataset = x_train.copy()
# train_dataset 에다가 y_train을 넣고 그래프를 보기 위함.
train_dataset.insert(0, 'WAGE', y_train)

_ = sns.pairplot(train_dataset, kind = 'reg', diag_kind = 'kde')

survey.data.info()

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ['RACE', 'OCCUPATION', 'SECTOR', 'MARR', 'UNION', 'SEX', 'SOUTH']
numerical_columns = ['EDUCATION', 'EXPERIENCE', 'AGE']

preprocessor = make_column_transformer(
    (OneHotEncoder(drop = 'if_binary'), categorical_columns), 
    remainder = 'passthrough'
)


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge 
from sklearn.compose import TransformedTargetRegressor

model = make_pipeline(
    preprocessor, 
    TransformedTargetRegressor(
        regressor = Ridge(alpha = 1e-10), func = np.log10, inverse_func = sp.special.exp10
    )
)

model.fit(x_train, y_train)

from sklearn.metrics import median_absolute_error

y_pred = model.predict(x_train)

mae = median_absolute_error(y_train, y_pred)
string_score = f'MAE on training set: {mae:.2f} $/hour'

y_pred = model.predict(x_test)
mae = median_absolute_error(y_test, y_pred)
string_score += f'\nMAE on test set: {mae:.2f} $/hour'

fig, ax = plt.subplots(figsize = (5, 5))
plt.scatter(y_test, y_pred)
ax.plot([0, 1], [0, 1], ls = '--', c = 'red')
plt.text(3, 20, string_score)
plt.title("Ridge model, small regularization")
plt.ylabel('Model predictions')
plt.xlabel('Truths')
plt.xlim([0, 27])
_ = plt.ylim([0, 27])


# Interpreting coefficients : scale matters 

x_train_preprocessed = pd.DataFrame(
    model.named_steps['columntransformer'].transform(x_train)
)

x_train_preprocessed.std(axis = 0).plot(kind = 'barh', figsize = (9, 7));
plt.title('Feature std. dev.')
plt.subplots_adjust(left = 0.3)

# Cross Validation ----
import numpy as np 
import sklearn.linear_model as linear_model

reg = linear_model.RidgeCV(alphas = np.logspace(-6, 6, 13))

x_data = [[0, 0], [0, 0], [1, 1]]
y_data = [0, 0.1, 1]

reg.fit(x_data, y_data)

reg.alpha_ # 0.01
