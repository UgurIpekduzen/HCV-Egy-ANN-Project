from keras.models import Sequential
from keras.layers import Dense
from keras import *
from termcolor import cprint
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from dataset import *

dataset = setDataFrame()
plot_corr(dataset)

# trainDataset = np.loadtxt("./input/HCV-Egy-Data-ANN-Train.csv", delimiter=",")
# testDataset = np.loadtxt("./input/HCV-Egy-Data-ANN-Test.csv", delimiter=",")

# print(trainDataset.shape[1])
# np.random.shuffle(trainDataset)
# xTrain = trainDataset[:len(trainDataset), 0:28]
# yTrain = trainDataset[:len(trainDataset), 28:29]
#
# # xValidation = trainDataset[100:200, 0:28]
# # yValidation = trainDataset[100:200, 28:29]
#
# xTest = testDataset[:len(testDataset), 0:28]
# yTest = testDataset[:len(testDataset), 28:29]
#
# # Sınıf değerlerinin binary değerlere çevrilmesi
# encoder = LabelEncoder()
# encoder.fit(yTrain)
# yTrain = encoder.transform(yTrain)
# yTrain = to_categorical(yTrain)
#
# print(yTrain)
#
# # encoder.fit(yValidation)
# # yValidation = encoder.transform(yValidation)
# # yValidation = to_categorical(yValidation)
#
# model = Sequential()
#
# model.add(Dense(16, activation="relu", input_dim=28))
# model.add(Dense(33, activation="relu"))
# model.add(Dense(33, activation="relu"))
# model.add(Dense(33, activation="relu"))
# model.add(Dense(4, activation="softmax"))
#
# model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adamax(learning_rate=0.00001) , metrics=['accuracy','mae'])
#
# # model.fit(xTrain, yTrain, validation_data=(xValidation, yValidation), batch_size=1000, shuffle=True, verbose=1, epochs=5000)
#
# model.fit(xTrain, yTrain, validation_split=0.1, batch_size=1000, shuffle=True, verbose=1, epochs=5000)
#
# scores = model.evaluate(xTrain, yTrain)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
# # Tahmin sonuçlarının decimal hale çevrilmesi
# binaryPredictions = model.predict(xTest)
# predictionsIndexes = np.argmax(binaryPredictions, axis=1)
# decimalPredictions = predictionsIndexes + 1
#
# dogru = 0
# yanlis = 0
# toplam_veri = len(yTest)
#
# for x, y in zip(decimalPredictions, yTest):
#     if x == y:
#         cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_green", attrs=['bold'])
#         dogru += 1
#     else:
#         cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_red", attrs=['bold'])
#         yanlis += 1
#
# print("\n", "-" * 150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Bilme Sayısı: ", dogru,
#       "\nYanlış Bilme Sayısı: ", yanlis,
#       "\nBaşarı Yüzdesi: ", str(int(100 * dogru / toplam_veri)) + "%", sep="")