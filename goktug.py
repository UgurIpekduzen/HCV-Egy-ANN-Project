from keras.models import Sequential
from keras.layers import Dense
from termcolor import cprint
from sklearn.model_selection import train_test_split
import numpy


dataset = numpy.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]])


X = dataset[:len(dataset), 0:3]
Y = dataset[:len(dataset), 3]

model = Sequential()

model.add(Dense(2, activation='relu', input_dim=3))
model.add(Dense(6, activation='softmax'))
model.add(Dense(6))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')

history = model.fit(X, Y, validation_split=0.2, epochs=250, verbose=0)

yeni_girdi = numpy.array([7, 8, 9])
yeni_girdi = yeni_girdi.reshape((1, 3))
tahminler = model.predict(yeni_girdi)
print(tahminler)

yeni_girdi = numpy.array([0, 1, 2])
yeni_girdi = yeni_girdi.reshape((1, 3))
tahminler = model.predict(yeni_girdi)

print(tahminler)