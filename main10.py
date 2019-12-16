from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn import model_selection
from keras.optimizers import *
from keras.metrics import *
from sklearn.utils import shuffle
from termcolor import cprint
from dataset import *

dataset = setDataFrame()

data = shuffle(dataset)

X = dataset.drop(['Baseline_Histological_Staging'], axis=1)
X = np.array(X)
Y = dataset['Baseline_Histological_Staging']

print("Repeating outputs are analyzing...")
stageNumbers = findRepeatedElements(Y)
stageNames = setStageNames(sorted(stageNumbers))

encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = to_categorical(Y)

# encoder.fit(Y2)
# Y2 = encoder.transform(Y2)
# Y2 = to_categorical(Y2)

trainX, testX, trainY, testY = model_selection.train_test_split(X, Y, test_size=0.231, random_state=0)

model = Sequential([
    Dense(100, activation="relu", input_dim=28),
    Dense(100, activation="tanh"),
    Dense(100, activation="tanh"),
    Dense(100, activation="tanh"),
    Dense(4, activation="sigmoid")
])

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=[mse, categorical_accuracy])
model.compile(loss='mse', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=64, verbose=1, epochs=1000)
scores = model.evaluate(trainX, trainY)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Tahmin sonuçlarının decimal hale çevrilmesi
binaryPredictions = model.predict(testX)
predictions = np.argmax(binaryPredictions, axis=1) + 1

targets = np.argmax(testY, axis=1) + 1

# predictionsIndexes = np.argmax(binaryPredictions, axis=1)
# decimalPredictions = predictionsIndexes + 1
#
# targetIndexes = np.argmax(testY, axis=1) + 1
# decimalTargets = targetIndexes + 1

dogru = 0
yanlis = 0
toplam_veri = len(targets)

for x, y in zip(predictions, targets):
    if x == y:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_green", attrs=['bold'])
        dogru += 1
    else:
        cprint("Tahmin: " + str(x) + " - Gerçek Değer: " + str(int(y)), "white", "on_red", attrs=['bold'])
        yanlis += 1

print("\n", "-" * 150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Bilme Sayısı: ", dogru,
      "\nYanlış Bilme Sayısı: ", yanlis,
      "\nBaşarı Yüzdesi: ", str(int(100 * dogru / toplam_veri)) + "%", sep="")

confusionMatrix = plot_cnf_matrix(predicted=setStageNames(predictions), target=setStageNames(targets),
                                  classes=stageNames
                                  , normalize=False)

# plot_roc(X_train=trainX, X_test=testX, Y_train= trainY, Y_test=testY, stage_names=stageNames)
plot_roc2(Y_test=testY, predictions=binaryPredictions, stage_names=stageNames)
