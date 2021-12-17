import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.datasets import fetch_openml 

# if as_frame -> load Data.frame 
survey = fetch_openml(data_id = 534, as_frame = True)

survey.target

x = survey.data[survey.feature_names]
# 다차원 배열을 1차원 배열로 평평하게 펴주는 ravel, flatten
y = survey.target.values.ravel()
y
