from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn import  model_selection
from keras.optimizers import *
from sklearn.utils import shuffle
from termcolor import cprint
from sklearn.metrics import plot_roc_curve


from dataset import *

dataset = setDataFrame()
print(dataset)

data = shuffle(dataset)

X = data.drop(['Baseline_Histological_Staging'], axis=1)
X = np.array(X)
Y = data['Baseline_Histological_Staging']
stageNumbers = findRepeatedElements(Y)
stageNames = setStageNames(sorted(stageNumbers))
nClasses = len(stageNames)

encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = to_categorical(Y)

# encoder.fit(Y2)
# Y2 = encoder.transform(Y2)
# Y2 = to_categorical(Y2)

trainX, testX, trainY, testY = model_selection.train_test_split(X, Y,test_size = 0.231, random_state = 0)

model = Sequential()

model.add(Dense(16, activation="relu", input_dim=28))
model.add(Dense(100, activation="tanh"))
# model.add(Dense(33, activation="relu"))
# model.add(Dense(33, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=128, shuffle=True, verbose=1, epochs=70)
scores = model.evaluate(trainX, trainY)
# falsePosRate = dict()
# truePosRate = dict()
# rocAuc = dict()
# for i in range(nClasses):
#     falsePosRate[i], truePosRate[i], _ = roc_curve(y[:, i], y_score[:, i])
#     roc_auc[i] = auc(falsePosRate[i], tpr[i])
# disp = plot_roc_curve(model, testX, testY)
# plt.show()


print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Tahmin sonuçlarının decimal hale çevrilmesi
predictions = model.predict(testX)
predictions = np.argmax(predictions, axis=1) + 1

targets = np.argmax(testY, axis=1) + 1
plot_cnf_matrix(predicted=setStageNames(predictions), target=setStageNames(targets),
                classes=stageNames
                ,normalize=False)

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