from keras.models import Sequential
from keras.layers import Dense
from termcolor import cprint
from sklearn.model_selection import train_test_split
import numpy

dataset = numpy.loadtxt("./input/test.csv", delimiter=",")

X = dataset[:1385, 0:27]
Y = dataset[:1385, 29]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size= 0.2, random_state= 0)

model = Sequential()

model.add(Dense(16, activation="relu", input_dim=27))
model.add(Dense(34, activation="softmax"))
model.add(Dense(5, activation="sigmoid"))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.fit(xTrain, yTrain, validation_data=(xTest, yTest), batch_size=32, shuffle=True, verbose=2, epochs=5)

scores = model.evaluate(X, Y)

test_verisi = dataset[len(xTrain): 1385, 0:27]

predictions = model.predict(test_verisi)

dogru = 0
yanlis = 0
toplam_veri = len(dataset[len(xTrain):1385, 27])

for x, y in zip(predictions, dataset[len(xTrain):1385, 29]):
    x = int(numpy.round(x[0]))
    if int(x) == y:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_green", attrs=['bold'])
        dogru += 1
    else:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_red", attrs=['bold'])
        yanlis += 1

print("\n", "-" * 150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Bilme Sayısı: ", dogru,
      "\nYanlış Bilme Sayısı: ", yanlis,
      "\nBaşarı Yüzdesi: ", str(int(100 * dogru / toplam_veri)) + "%", sep="")
