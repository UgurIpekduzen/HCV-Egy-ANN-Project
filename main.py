from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from termcolor import cprint
from dataset import *

#  Veri setinin hazırlanması
epochValue = 10000
dataset = setDataFrame()
# plot_corr(dataset)
dataset = shuffle(dataset)

# Veri setinin giriş ve çıkışlara ayrılması
X = dataset.drop(['BHS'], axis=1)
X = np.array(X)
Y = dataset['BHS']

# Veri setinindeki çıkışların isimlendirilmesi
print("\nRepeating outputs are analyzing...\n")
stageNumbers = findRepeatedElements(Y)
stageNames = setStageNames(sorted(stageNumbers))

# Hastalık seviyelerinin binary formata çevrilmesi
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = to_categorical(Y)

# Veri setinin kümelere ayrılması.
trainX, testX, trainY, testY = model_selection.train_test_split(X, Y, test_size=0.15, random_state=0)

# Modelin katmanlara ayrılması
model = Sequential([
    Dense(32, activation="tanh", input_dim=28),
    Dense(4, activation="sigmoid")
])

model.compile(loss='mse', optimizer=RMSprop(lr=0.1), metrics=['accuracy'])

# Modelin eğitilmesi ve train accuracy'nin ekrana basılması
history = model.fit(trainX, trainY, batch_size=1024, validation_split=0.15, verbose=1, epochs=epochValue)
scores = model.evaluate(trainX, trainY)

# Accuracy sonucunun ekrana basılması
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Tahmin sonuçlarının decimal hale çevrilmesi
binaryPredictions = model.predict(testX)
predictions = np.argmax(binaryPredictions, axis=1) + 1
targets = np.argmax(testY, axis=1) + 1

dogru = 0
yanlis = 0
toplam_veri = len(targets)

# Sırasıyla bütün test verilerinin tahmin sonucu ve gerçek sonucun kıyaslanarak ekrana basılması
for x, y in zip(predictions, targets):
    if x == y:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_green", attrs=['bold'])
        dogru += 1
    else:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_red", attrs=['bold'])
        yanlis += 1

#  Tahminlerin istatistiksel olarak doğruluk sonuçları
print("\n", "-" * 150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Bilme Sayısı: ", dogru,
      "\nYanlış Bilme Sayısı: ", yanlis,
      "\nBaşarı Yüzdesi: ", str(int(100 * dogru / toplam_veri)) + "%", sep="")

# Confusion Matrix'in hazırlanması ve çizdirilmesi
confusionMatrix = plot_cnf_matrix(predicted=setStageNames(predictions), target=setStageNames(targets),
                                  stages=stageNames
                                  , normalize=False)

# ROC Curve grafiğinin hazırlanması ve çizdirilmesi

# plot_roc(X_train=trainX, X_test=testX, Y_train= trainY, Y_test=testY, stage_names=stageNames)
plot_roc2(Y_test=testY, predictions=binaryPredictions, stage_names=stageNames)

# Eğitim ve doğrulama hatalarının grafiğinin hazırlanması ve çizdirilmesi
plot_train_and_val_loss(history=history, epoch=epochValue)
