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



X = dataset[:,0:28]
Y = dataset[:, 28:29]
print(X.shape)
print(Y.shape)

model = Sequential()

model.add(Dense(400, input_dim=28, activation='relu'))
# İkinci katmanımızda 12 yapay sinir hücresi.
model.add(Dense(1, activation='sigmoid'))

# Modelimizi derliyoruz.
sgd = SGD(lr=0.9, decay=1e-6, momentum=1, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

model.fit(X, Y, epochs=50, validation_split=0.13)

tahmin = np.array([39,2,29,1,2,1,2,1,1,2,7136,4625248.00,10,211363.00,70,102,76.00,58,111,95,58,25,993940,992652,96482,334897,762760,15]).reshape(1,28)
print(model.predict(tahmin))
