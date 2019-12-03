from keras.models import Sequential
from keras.layers import Dense
from termcolor import cprint
from sklearn.model_selection import train_test_split
import numpy as np

dataset = np.loadtxt("./input/test.csv", delimiter=",")

X = dataset[:1385, 0:27]
Y = dataset[:1385, 28]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=0)

targetClassified = np.zeros(shape=(yTrain.shape[0], 5))
print(yTrain.shape)
for i in range(yTrain.shape[0]):
    if yTrain[i] == 0:
        targetClassified[i] = np.array([1, 0, 0, 0, 0])
    elif yTrain[i] == 1:
        targetClassified[i] = np.array([0, 1, 0, 0, 0])
    elif yTrain[i] == 2:
        targetClassified[i] = np.array([0, 0, 1, 0, 0])
    elif yTrain[i] == 3:
        targetClassified[i] = np.array([0, 0, 0, 1, 0])
    elif yTrain[i] == 4:
        targetClassified[i] = np.array([0, 0, 0, 0, 1])

model = Sequential()

model.add(Dense(16, activation="relu", input_dim=27))
model.add(Dense(20, activation="softmax"))
model.add(Dense(5, activation="sigmoid"))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.fit(xTrain, targetClassified, batch_size=64, shuffle=True, verbose=2, epochs=5)

test_verisi = dataset[len(xTrain): 1385, 0:27]

predictions = model.predict(test_verisi)

predictionResults = np.zeros(shape=(predictions.shape[0], 1))
for i in range(predictions.shape[0]):
    if np.array_equal(predictions[i], np.array([1, 0, 0, 0, 0])):
        predictionResults[i] = 0
    elif np.array_equal(predictions[i], np.array([0, 1, 0, 0, 0])):
        predictionResults[i] = 1
    elif np.array_equal(predictions[i], np.array([0, 0, 1, 0, 0])):
        predictionResults[i] = 2
    elif np.array_equal(predictions[i], np.array([0, 0, 0, 1, 0])):
        predictionResults[i] = 3
    elif np.array_equal(predictions[i], np.array([0, 0, 0, 0, 1])):
        predictionResults[i] = 4
    else:
        predictionResults[i] = -1

dogru = 0
yanlis = 0
toplam_veri = len(dataset[len(xTrain):1385, 27])

for x, y in zip(predictionResults, dataset[len(xTrain):1385, 28]):
    x = int(np.round(x[0]))
    if int(x) == y:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_green", attrs=['bold'])
        dogru += 1
    else:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_red", attrs=['bold'])
        yanlis += 1

print("\n", "-" * 150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Bilme Sayısı: ", dogru,
      "\nYanlış Bilme Sayısı: ", yanlis,
      "\nBaşarı Yüzdesi: ", str(int(100 * dogru / toplam_veri)) + "%", sep="")
