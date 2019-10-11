"""Load and visualize the iris dataset"""
import sklearn.datasets as datasets


iris_data = datasets.load_iris()


if __name__ == '__main__':
    for i in iris_data:
        print('\n', i)
        print(iris_data[i])
