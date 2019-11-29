# Kullanacağımız ysa modeli için.
from keras.models import Sequential

# YSA modelimizde katmanlar oluşturabilmek için.
from keras.layers import Dense

# Çıktımızı terminalden aldığımızda sonuçları renklendiriyoruz. Yanlışlar kırmızı, doğrular yeşil. Bunu kullanmasanızda olur yani.
from termcolor import cprint

# YSA matrislerle çalıştığı için numpy olmazsa olmaz.
import numpy

dataset = numpy.loadtxt("./input/HCV-Egy-Data.csv", delimiter=",")

X = dataset[:600, 0:27]
Y = dataset[:600, 27:29]
print(X)
print(Y)

model = Sequential()

model.add(Dense(20, input_dim=27, init='uniform', activation='relu'))
# İkinci katmanımızda 12 yapay sinir hücresi.
model.add(Dense(12, init='uniform', activation='relu'))

# Üçüncü katmanımızda 8 yapay hücremiz var.
model.add(Dense(8, init='uniform', activation='sigmoid'))

# Dördüncü katmanımızda 2 yapay hücremiz var. Yani çıkışımız.
model.add(Dense(2, init='uniform', activation='sigmoid'))

# Modelimizi derliyoruz.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=2)

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# 96 adet test verimizin sadece 27 girdisini sisteme veriyoruz.
test_verisi = dataset[600:696, 0:27]

# Sistemin verdiğimiz değerlerden yola çıkarak kişinin diyabet hastası olup olmadığını tahmin ediyor.
predictions = model.predict(test_verisi)

dogru = 0
yanlis = 0
toplam_veri = len(dataset[600:696, 27])

for x, y in zip(predictions, dataset[600:696, 27]):
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
