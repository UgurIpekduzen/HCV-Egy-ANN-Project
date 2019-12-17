# Kullanacağımız ysa modeli için.
from keras.models import Sequential

# YSA modelimizde katmanlar oluşturabilmek için.
from keras.layers import Dense

# Çıktımızı terminalden aldığımızda sonuçları renklendiriyoruz. Yanlışlar kırmızı, doğrular yeşil. Bunu kullanmasanızda olur yani.
from termcolor import cprint

# YSA matrislerle çalıştığı için numpy olmazsa olmaz.
import numpy

dataset = numpy.loadtxt("./input/HCV-Egy-Data.csv", delimiter=",")

training_data_count = 100
testing_data_count = 25
girdi_sayisi=27
cikti_sayisi=1

X = dataset[:training_data_count, 0:girdi_sayisi]
Y = dataset[:training_data_count, girdi_sayisi+cikti_sayisi]

model = Sequential()

model.add(Dense(14, input_dim=girdi_sayisi, activation='relu'))

model.add(Dense(14, activation='softmax'))

# Dördüncü katmanımızda 1 yapay hücremiz var. Yani çıkışımız.
model.add(Dense(1, activation='sigmoid'))

# Modelimizi derliyoruz.
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, validation_split=0.2, epochs=150, verbose=2)

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# 96 adet test verimizin sadece 27 girdisini sisteme veriyoruz.
test_verisi = dataset[training_data_count:training_data_count+testing_data_count, 0:girdi_sayisi]

# Sistemin verdiğimiz değerlerden yola çıkarak kişinin hepatit c hastası olup olmadığını tahmin ediyor.
predictions = model.predict(test_verisi)

dogru = 0
yanlis = 0
toplam_veri = len(dataset[training_data_count:training_data_count+testing_data_count, girdi_sayisi])

for x, y in zip(predictions, dataset[training_data_count:training_data_count+testing_data_count, girdi_sayisi+cikti_sayisi]):
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
