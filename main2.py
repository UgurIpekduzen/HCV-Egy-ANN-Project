# Kullanacağımız ysa modeli için.
from keras.models import Sequential

# YSA modelimizde katmanlar oluşturabilmek için.
from keras.layers import Dense
from keras.optimizers import SGD
# Çıktımızı terminalden aldığımızda sonuçları renklendiriyoruz. Yanlışlar kırmızı, doğrular yeşil. Bunu kullanmasanızda olur yani.
from termcolor import cprint

# YSA matrislerle çalıştığı için numpy olmazsa olmaz.
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dataset = np.loadtxt("./input/HCV-Egy-Data.csv", delimiter=",")

training_data_count = 800
testing_data_count = 25

input = dataset[:training_data_count, 0: 27]
target = dataset[:training_data_count, 28: 29]


targetClassified = np.zeros(shape=(target.shape[0], 5))
print(target.shape)
for i in range(target.shape[0]):
    if target[i] == 1:
        targetClassified[i] = np.array([0, 1, 0, 0, 0])
    elif target[i] == 2:
        targetClassified[i] = np.array([0, 0, 1, 0, 0])
    elif target[i] == 3:
        targetClassified[i] = np.array([0, 0, 0, 1, 0])
    elif target[i] == 4:
        targetClassified[i] = np.array([0, 0, 0, 0, 1])

print(targetClassified)

model = Sequential()

model.add(Dense(400, input_dim=27, activation='relu'))
model.add(Dense(400, activation='relu'))
# İkinci katmanımızda 12 yapay sinir hücresi.
model.add(Dense(5, activation='sigmoid'))

# Modelimizi derliyoruz.
sgd = SGD(lr=0.9, decay=1e-6, momentum=1, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

model.fit(input, targetClassified, epochs=50, validation_split=0.13, verbose=2)

test_verisi = dataset[training_data_count:training_data_count+testing_data_count, 0:27]

predictions = model.predict(test_verisi)

predictionResults = np.zeros(shape=(predictions.shape[0], 1))
for i in range(predictions.shape[0]):
    if np.array_equal(predictions[i], np.array([0, 1, 0, 0, 0])):
        predictionResults[i] = 1
    elif np.array_equal(predictions[i], np.array([0, 0, 1, 0, 0])):
        predictionResults[i] = 2
    elif np.array_equal(predictions[i], np.array([0, 0, 0, 1, 0])):
        predictionResults[i] = 3
    elif np.array_equal(predictions[i], np.array([0, 0, 0, 0, 1])):
        predictionResults[i] = 4
    else:
        predictionResults[i] = 0

dogru = 0
yanlis = 0
toplam_veri = len(dataset[training_data_count:training_data_count+testing_data_count, 27])

for x, y in zip(predictionResults, dataset[training_data_count:training_data_count+testing_data_count, 28:29]):
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




