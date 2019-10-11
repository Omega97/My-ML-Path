"""This simple model predicts the class of the flowers"""
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from random import randrange


# load dataset
iris_dataset = load_iris()

# extract data and labels
X = iris_dataset['data']
Y = iris_dataset['target']
Y = [1 if i == 0 else 0 for i in Y]

# build the model
Model = LinearRegression()

# training
Model.fit(X, Y)
print('score =', Model.score(X, Y), '\n')


def predictor(model, x):
    return 1 if model.predict([x]) > .5 else 0


accuracy = sum([predictor(Model, X[i]) == Y[i] for i in range(len(X))]) / len(Y)
print('accuracy =', round(accuracy * 100, 1), '%', '\n')

# make predictions
for _ in range(20):
    x_ = X[randrange(len(X))]
    print(x_, end=' ')
    if predictor(Model, x_):
        print('setosa')
    else:
        print(' - ')
