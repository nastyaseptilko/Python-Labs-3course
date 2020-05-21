import  sys
print("версия Python: {}".format(sys.version))

import numpy as np
print("версия NumPy: {}".format(np.__version__))

import pandas as pd

import scipy as sp
print("версия SciPy: {}".format(sp.__version__))

import sklearn
print("версия scikit-learn: {}".format(sklearn.__version__))

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Названия ответов: {}".format(iris_dataset['target_names']))
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))
print("Тип массива data: {}".format(type(iris_dataset['data'])))
print("Форма массива data: {}".format(iris_dataset['data'].shape))
print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))
print("Тип массива target: {}".format(type(iris_dataset['target'])))
print("Форма массива target: {}".format(iris_dataset['target'].shape))
print("Ответы:\n{}".format(iris_dataset['target']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)
print("форма массива X_train: {}".format(X_train.shape))
print("форма массива y_train: {}".format(y_train.shape))
print("форма массива X_test: {}".format(X_test.shape))
print("форма массива y_test: {}".format(y_test.shape))

import mglearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
    hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()


