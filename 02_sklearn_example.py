"""Simple implementation of the sklearn linear regression"""
from sklearn.linear_model import LinearRegression
import numpy as np
from random import randrange


# load data
m1, m2, q = 1, 2, 3
X = np.array([[randrange(5), randrange(5)] for _ in range(10)])
Y = np.dot(X, np.array([m1, m2])) + q

# build model
reg = LinearRegression()

# training
reg.fit(X, Y)
reg.score(X, Y)

# results
print('regression coefficient =', reg.coef_)
print('regression intercept =', reg.intercept_)
v = [3, 5]
print('predict', v)
print('regression prediction =', reg.predict(np.array([v]))[0])
