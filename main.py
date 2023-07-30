import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Version :", tf.__version__)
fashion = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion.load_data()

class_items = [
"T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot"
]
print("First Test :", class_items[y_train[0]])
plt.clf()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
SNN = keras.models.Sequential()
SNN.add(keras.layers.Flatten(input_shape=[28, 28]))
SNN.add(keras.layers.Dense(300, "relu"))
SNN.add(keras.layers.Dense(100, "relu"))
SNN.add(keras.layers.Dense(10, activation="softmax"))
print(SNN.summary())
Hidden1 = SNN.layers[1]
Hidden2 = SNN.layers[2]
weights, biases = Hidden1.get_weights()
SNN.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = SNN.fit(X_train, y_train, epochs=5)
print(SNN.evaluate(X_test, y_test))
predict_y = SNN.predict(X_test)
print("\n\n", predict_y.round(2))
predict_objects = np.argmax(predict_y, axis=1)
print("\n\n", predict_objects)
np.array(class_items)[predict_objects]

def printPredict(number):
    plt.figure(1); plt.clf()
    plt.imshow(X_test[number], cmap='gray')
    plt.title("The Prediction For This Item Is : " + str(class_items[predict_objects[number]]) + "(" + str(class_items[y_test[number]] + ")"))
    plt.pause(3)
for i in range(10000):
    printPredict(i)
plt.show()
