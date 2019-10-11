import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import argmax


# load data
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# load model
model = tf.keras.models.load_model('mnist_model.model')


# make predictions
indexes = [i for i in range(18)]
predictions = model.predict([[x_test[i] for i in indexes]])
predictions = [argmax(i) for i in predictions]


# plot
for i in indexes:
    plt.subplot(len(indexes)//6, len(indexes)//3, 1+i)
    plt.imshow(x_test[i], cmap='binary')
    plt.title(str(predictions[i]) + ' (' + str(y_test[i]) + ')')
plt.show()
