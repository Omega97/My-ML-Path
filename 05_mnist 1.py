import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import argmax


# loas data
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))     # number of classes

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=2)

# evaluate model
val_loss, val_acc = model.evaluate(x_test, y_test)
print('loss =', val_loss)
print('acc =', val_acc)

# save the model
model.save('mnist_model.model')

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
