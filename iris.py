from keras.models import Sequential
from keras.layers import Dense
from termcolor import cprint
from sklearn.model_selection import train_test_split
import numpy

dataset = numpy.loadtxt("./input/iris.csv", delimiter=",")

X = dataset[:len(dataset), 0:4]
Y = dataset[:len(dataset), 4]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=0)
xTest = xTrain
yTest = yTrain
model = Sequential([
    Dense(3, activation="tanh", input_dim=4),
    Dense(1, activation="sigmoid")
])

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.fit(xTrain, yTrain, batch_size=32, shuffle=True, verbose=0, epochs=50)

scores = model.evaluate(X, Y)

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
