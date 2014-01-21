__author__ = 'nastra'

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import numpy as np


def loadBostonData():
    """
    Data Set Information:

    Concerns housing values in suburbs of Boston.


    Attribute Information:

    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000's
    """
    boston = load_boston()
    return boston.data, boston.target, boston.DESCR, boston.feature_names


def plotFeatures(data, target, xlabel, ylabel):
    x = data[:, 5]
    x = np.array([[v] for v in x])
    plt.scatter(data[:, 5], target, color='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    slope, res, _, _ = np.linalg.lstsq(x, y)
    plt.plot([0, data[:, 5].max() + 1], [0, slope * (data[:, 5].max() + 1)], '-', lw=4)
    plt.savefig('charts/Figure1.png', dpi=150)


if __name__ == "__main__":
    X, y, descr, featureNames = loadBostonData()
    #classifier = LinearRegression(normalize=True, fit_intercept=True)
    kfold = KFold(len(X), n_folds=10, shuffle=True)
    
    # normalized Rigde seems to do best in this example
    for name, method in [
        ('elastic-net(.5)', ElasticNet(fit_intercept=True, alpha=0.5, normalize=True)),
        ('lasso(.5)', Lasso(fit_intercept=True, alpha=0.5, normalize=True)),
        ('ridge(.5)', Ridge(fit_intercept=True, alpha=0.5, normalize=True)),
    ]:
        squaredErrors = 0
        for train, test in kfold:
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]
            y_test = y[test]

            #classifier.fit(X_train, y_train)
            method.fit(X_train, y_train)
            #prediction = classifier.predict(X_test)
            prediction = method.predict(X_test)
            squaredError = mean_squared_error(y_test, prediction)
            squaredErrors += squaredError

        rmse = np.sqrt(squaredErrors / len(kfold))
        print("=" * 50)
        print('Method: {}'.format(name))
        print("Root Mean Squared Error: " + str(rmse))
        print("=" * 50)