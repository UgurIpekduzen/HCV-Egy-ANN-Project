from keras.models import Sequential
from keras.layers import Dense
from termcolor import cprint
import numpy

dataset = numpy.loadtxt("./input/HCV-Egy-Data.csv", delimiter=",")

training_data_count = 1000
testing_data_count = 200

X = dataset[:training_data_count, 0:27]
Y = dataset[training_data_count, 28]

model = Sequential()

model.add(Dense(14, activation="relu", input_dim=27))
model.add(Dense(1, activation="softmax"))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, validation_split=0.2, epochs=150, verbose=2)

scores = model.evaluate(X, Y)

print(scores)

test_verisi = dataset[training_data_count: training_data_count+testing_data_count, 0:27]

predictions = model.predict(test_verisi)