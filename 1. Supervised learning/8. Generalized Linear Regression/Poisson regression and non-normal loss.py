import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model 

from sklearn.datasets import fetch_openml

df = fetch_openml(data_id = 41214, as_frame = True).frame 
df.columns
df['Frequency'] = df['ClaimNb'] / df['Exposure']

print(
    f'Average Frequency = {np.average(df["Frequency"], weights = df["Exposure"])}'
)

# pipeline -> make_pipeline
# preprocessing -> FunctionTransformer
# compose -> ColumnTransformer

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer, 
    OneHotEncoder, 
    StandardScaler, 
    KBinsDiscretizer
    )
from sklearn.compose import ColumnTransformer


log_scale_transformer = make_pipeline(
    FunctionTransformer(np.log, validate = False), StandardScaler()
)

linear_model_preprocessor = ColumnTransformer(
    [
        ("passthrough-numeric", 'passthrough', ['BonusMalus']), 
        ('ninned_numeric', KBinsDiscretizer(n_bins = 10), ['VehAge', 'DrivAge']), 
        ("log_scaled_numeric", log_scale_transformer, ['Density']), 
        (
            'onehot_categorical', 
            OneHotEncoder(), 
            ['VehBrand', 'VehPower', 'VehGas', 'Region', 'Area']
        )
    ], 
    remainder = 'drop',
)

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size = 0.33, random_state = 42)

dummy = Pipeline(
    [
        ('preprocessor', linear_model_preprocessor), 
        ('regressor', DummyRegressor(strategy = 'mean')),
    ]
).fit(df_train, df_train['Frequency'], regressor__sample_weight = df_train['Exposure'])


from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_poisson_deviance
)

def score_estimator(estimator, df_test):
    """Score an estimator on the test set."""
    y_pred = estimator.predict(df_test)

    
    print(
        'MSE : %.3f' % mean_squared_error(df_test['Frequency'], y_pred, sample_weight = df_test['Exposure'])
    )
    
    print(
        'MAE : %.3f' % mean_absolute_error(df_test['Frequency'], y_pred, sample_weight = df_test['Exposure'])
    )
    
    # Ignore non-positive predictions, as they are invalid for the Poisson deviance.
    
    # any() -> 전달받은 자료형의 element 중 하나라도 True 일 경우 True를 반환해줌.
    mask = y_pred > 0 
    if (~mask).any() :
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print('WARNING : Estimator yields invalid, non-positive predictions')
    
    print(
        'mean Poisson deviance : %.3f'
        % mean_poisson_deviance(
            df_test['Frequency'][mask], 
            y_pred[mask], 
            sample_weight = df_test['Exposure'][mask], 
        )
    )
    
print('Constant mean frequency evaluation:')
score_estimator(dummy, df_test)


# Generalized linear models 
from sklearn.linear_model import Ridge 

ridge_glm = Pipeline(
    [
        ('preprocessor', linear_model_preprocessor), 
        ('regressor', Ridge(alpha = 1e-6))
    ]
).fit(df_train, df_train['Frequency'], regressor__sample_weight = df_train['Exposure'])

print('Ridge evaluation:')
score_estimator(ridge_glm, df_test)


# PoissonRegressor
from sklearn.linear_model import PoissonRegressor

n_samples = df_train.shape[0]

poisson_glm = Pipeline(
    [
        ('preprocessor', linear_model_preprocessor), 
        ('regressor', PoissonRegressor(alpha = 1e-12, max_iter = 300))
    ]
)

poisson_glm.fit(
    df_train, df_train['Frequency'], regressor__sample_weight = df_train['Exposure']
)

print("PoissonRegressor evaluation")
score_estimator(poisson_glm, df_test)

# Gradient Boosting Regression Trees for Poisson regression
from sklearn.ensemble import GradientBoostingRegressor # HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

tree_preprocessor = ColumnTransformer(
    [
        (
            "categorical", 
            OrdinalEncoder(), 
            ['VehBrand', 'VehPower', 'VehGas', 'Region', 'Area']
        )
    ]
)

poisson_gbrt = Pipeline(
    [
        ('preprocessor', tree_preprocessor), 
        ('regressor', GradientBoostingRegressor(max_leaf_nodes=128, loss = 'ls'))
    ]
)

poisson_gbrt.fit(df_train, df_train['Frequency'], regressor__sample_weight = df_train['Exposure'])

print('Poisson Gradient Boosted Trees evaluation:')
score_estimator(poisson_gbrt, df_test)


# graph 

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (16, 6), sharey = True)
fig.subplots_adjust(bottom = 0.2)

n_bins = 20
for row_idx, label, df in zip(range(2), ['train', 'test'], [df_train, df_test]):
    df['Frequency'].hist(bins = np.linspace(-1, 30, n_bins), ax = axes[row_idx, 0])
    
    axes[row_idx, 0].set_title('Data')
    axes[row_idx, 0].set_yscale('log')
    axes[row_idx, 0].set_xlabel('y (observed Frequency)')
    axes[row_idx, 0].set_ylim([1e1, 5e5])
    axes[row_idx, 0].set_ylabel(label + ' samples')
    
    for idx, model in enumerate([ridge_glm, poisson_glm, poisson_gbrt]):
        y_pred = model.predict(df)
        
        pd.Series(y_pred).hist(bins = np.linspace(-1, 4, n_bins), ax = axes[row_idx, idx + 1])
        axes[row_idx, idx + 1].set(title = model[-1].__class__.__name__, 
                                   yscale = 'log', 
                                   xlabel='y_pred (predicted expected Frequency')
plt.show()

from sklearn.linear_model import LinearRegression


# Evaluation of the calibration of predictions 

from sklearn.utils import gen_even_slices

def _mean_frequency_by_risk_group(y_true, y_pred, sample_weight = None, n_bins = 100):
    
    idx_sort = np.argsort(y_pred)
    bin_centers = np.arange(0, 1, 1 / n_bins) + 0.5 / n_bins
    y_pred_bin = np.zeros(n_bins)
    y_true_bin = np.zeros(n_bins)
    
    for n, sl in enumerate(gen_even_slices(len(y_true), n_bins)):
        weights = sample_weight[idx_sort][sl]
        y_pred_bin[n] = np.average(y_pred[idx_sort][sl], weights = weights)
        y_true_bin[n] = np.average(y_true[idx_sort][sl], weights = weights)
    return bin_centers, y_true_bin, y_pred_bin

print(f'Actual number of claims: {df_test["ClaimNb"].sum()}')
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 8))
plt.subplots_adjust(wspace = 0.3)

for axi, model in zip(ax.ravel(), [ridge_glm, poisson_glm, poisson_gbrt, dummy]):
    y_pred = model.predict(df_test)
    y_true = df_test['Frequency'].values 
    exposure = df_test['Exposure'].values 
    q, y_true_seg, y_pred_seg = _mean_frequency_by_risk_group(
        y_true, y_pred, sample_weight = exposure, n_bins = 10
    )
    
    print(f'Predicted number of claims by {model[-1]}: {np.sum(y_pred * exposure):.1f}')
    
    axi.plot(q, y_pred_seg, marker = 'x', linestyle = '--', label = 'predictions')
    axi.plot(q, y_true_seg, marker = 'o', linestyle = '--', label = 'observations')
    axi.set_xlim(0, 1.0)
    axi.set_ylim(0, 0.5)
    axi.set(
        title = model[-1], 
        xlabel = 'Fraction of samples sorted by y_pred', 
        ylabel = 'Mean Frequency (y_pred)',
    )
    axi.legend()
plt.tight_layout()
plt.show()