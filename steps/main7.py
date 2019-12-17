from keras.models import Sequential
from keras.layers import Dense
from termcolor import cprint
import numpy

trainDataset = numpy.loadtxt("./input/HCV-Egy-Data-ANN-Train.csv", delimiter=",")
testDataset = numpy.loadtxt("./input/HCV-Egy-Data-ANN-Test.csv", delimiter=",")

numpy.random.shuffle(trainDataset)
xTrain = trainDataset[:len(trainDataset), 0:28]
yTrain = trainDataset[:len(trainDataset), 28]

xValidation = trainDataset[100:200, 0:28]
yValidation = trainDataset[100:200, 28]

xTest = testDataset[:len(testDataset), 0:28]
yTest = testDataset[:len(testDataset), 28]

model = Sequential()

model.add(Dense(6, activation="relu", input_dim=28))
model.add(Dense(1,activation="softmax"))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.fit(xTrain, yTrain, validation_data=(xValidation, yValidation), batch_size=32, shuffle=True, verbose=0, epochs=50)

scores = model.evaluate(xTrain, yTrain)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(xTest)
print(predictions[0])
print(predictions[1])
dogru = 0
yanlis = 0
toplam_veri = len(yTest)

for x, y in zip(predictions, yTest):
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
