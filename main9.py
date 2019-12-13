from keras.models import Sequential
from keras.layers import Dense
from termcolor import cprint
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


trainDataset = np.loadtxt("./input/HCV-Egy-Data-ANN-Train.csv", delimiter=",")
testDataset = np.loadtxt("./input/HCV-Egy-Data-ANN-Test.csv", delimiter=",")

np.random.shuffle(trainDataset)
xTrain = trainDataset[:len(trainDataset), 0:28]
yTrain = trainDataset[:len(trainDataset), 28:29]

xValidation = trainDataset[100:200, 0:28]
yValidation = trainDataset[100:200, 28:29]

xTest = testDataset[:len(testDataset), 0:28]
yTest = testDataset[:len(testDataset), 28:29]

# Sınıf değerlerinin binary değerlere çevrilmesi
encoder = LabelEncoder()
encoder.fit(yTrain)
yTrain = encoder.transform(yTrain)
yTrain = to_categorical(yTrain)

encoder.fit(yValidation)
yValidation = encoder.transform(yValidation)
yValidation = to_categorical(yValidation)

# encoder.fit(yTest)
# yTest = encoder.transform(yTest)
# yTest = to_categorical(yTest)

model = Sequential([
    Dense(16, activation="relu", input_dim=28),
    Dense(32, activation="tanh"),
    Dense(4, activation="sigmoid")
])

model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

model.fit(xTrain, yTrain, validation_data=(xValidation, yValidation), batch_size=32, shuffle=True, verbose=1, epochs=70)

scores = model.evaluate(xTrain, yTrain)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Tahmin sonuçlarının decimal hale çevrilmesi
binaryPredictions = model.predict(xTest)
predictionsIndexes = np.argmax(binaryPredictions, axis=1)
decimalPredictions = predictionsIndexes + 1

dogru = 0
yanlis = 0
toplam_veri = len(yTest)

for x, y in zip(decimalPredictions, yTest):
    if x == y:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_green", attrs=['bold'])
        dogru += 1
    else:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_red", attrs=['bold'])
        yanlis += 1

print("\n", "-" * 150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Bilme Sayısı: ", dogru,
      "\nYanlış Bilme Sayısı: ", yanlis,
      "\nBaşarı Yüzdesi: ", str(int(100 * dogru / toplam_veri)) + "%", sep="")