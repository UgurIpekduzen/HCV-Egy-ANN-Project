import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, auc, roc_curve
import itertools

# CSV dosyasında yer alan düzensiz satır ve sütunların tek tek parçalı bir şekilde aktarılması
def parseRowsAndColumns():
    rawData = np.loadtxt('./input/HCV-Egy-Data.csv', delimiter=",")
    splittedData = []
    for i in range(rawData.shape[0]):
        column = []
        for j in range(rawData.shape[1]):
            column.append(rawData[i][j])
        splittedData.append(column)
    return splittedData

# Parçalanmış satır ve sütun verilerinin pandas sorgularına uygun hale getirilmesi
def setDataFrame():
    dataColumns = parseRowsAndColumns()
    patientDict = {"data":[]}

    for i in range(len(dataColumns)):
        patient = {}
        patient['Age'] = int(dataColumns[i][0])
        patient['Gender'] = int(dataColumns[i][1])
        patient['BMI'] = int(dataColumns[i][2])
        patient['Fever'] = int(dataColumns[i][3])
        patient['Nausea/Vomting'] = int(dataColumns[i][4])
        patient['Headache'] = int(dataColumns[i][5])
        patient['Diarrhea'] = int(dataColumns[i][6])
        patient['Fatigue'] = int(dataColumns[i][7])
        patient['Bone_Ache'] = int(dataColumns[i][8])
        patient['Jaundice'] = int(dataColumns[i][9])
        patient['Epigastria_Pain'] = int(dataColumns[i][10])
        patient['WBC'] = int(dataColumns[i][11])
        patient['RBC'] = int(dataColumns[i][12])
        patient['HGB'] = int(dataColumns[i][13])
        patient['Plat'] = int(dataColumns[i][14])
        patient['AST_1'] = int(dataColumns[i][15])
        patient['ALT_1'] = int(dataColumns[i][16])
        patient['ALT_4'] = int(dataColumns[i][17])
        patient['ALT_12'] = int(dataColumns[i][18])
        patient['ALT_24'] = int(dataColumns[i][19])
        patient['ALT_36'] = int(dataColumns[i][20])
        patient['ALT_48'] = int(dataColumns[i][21])
        patient['RNA_Base'] = int(dataColumns[i][22])
        patient['RNA_4'] = int(dataColumns[i][23])
        patient['RNA_12'] = int(dataColumns[i][24])
        patient['RNA_EOT'] = int(dataColumns[i][25])
        patient['RNA_EF'] = int(dataColumns[i][26])
        patient['BHG'] = int(dataColumns[i][27])
        patient['BHS'] = int(dataColumns[i][28])
        patientDict['data'].append(patient)

    patientDataFrame = DataFrame.from_dict(patientDict['data'])

    return patientDataFrame

# Tamsayı şeklinde tutulan hastalık seviyelerinin isimlendirilmesi
def setStageNames(stages):
    stageNames = []

    for item in stages:
        stageName = ""
        if item == 1:
            stageName = "F1"
        elif item == 2:
            stageName = "F2"
        elif item == 3:
            stageName = "F3"
        elif item == 4:
            stageName = "F4"
        else:
            stageName = "F0"
        stageNames.append(stageName)
    return stageNames

# Veri setinde tanımlanmış hastalık seviyelerinin çıkarılması
def findRepeatedElements(x):
    _size = len(x)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if x[i] == x[j] and x[i] not in repeated:
                repeated.append(x[i])
    return repeated

# Korelasyon matrisinin çizdirilmesi
def plot_corr(df):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df.corr(), cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix')
    plt.show()

    print("Correlation Matrix is completed")

# Confusion matrisinin çizdirilmesi
def plot_cnf_matrix(target, predicted, stages, normalize=False):
    title = ""
    titleOptions = ["Confusion Matrix with Normalization", "Confusion Matrix without Normalization"]
    plt.figure()
    cnfMatrix = confusion_matrix(y_true=target, y_pred=predicted, labels=stages)
    np.set_printoptions(precision=2)
    cmap = plt.cm.Blues

    #Boolean değerine göre değerleri normalize eder
    if normalize:
        cnfMatrix = cnfMatrix.astype('float') / cnfMatrix.sum(axis=1)[:, np.newaxis]
        title = titleOptions[0]
    else:
        title = titleOptions[1]
    print(title)

    print(cnfMatrix)

    plt.imshow(cnfMatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(stages))
    plt.xticks(tick_marks, stages, rotation=45)
    plt.yticks(tick_marks, stages)

    fmt = '.2f' if normalize else 'd'
    thresh = cnfMatrix.max() / 2.
    for i, j in itertools.product(range(cnfMatrix.shape[0]), range(cnfMatrix.shape[1])):
        plt.text(j, i, format(cnfMatrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnfMatrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    print("Confusion Matrix is completed")

# ROC eğrisinin OneVsRestClassifier kullanılarak çizdirilmesi
def plot_roc(X_train, X_test, Y_train, Y_test, stage_names):
    n_stages = len(stage_names)

    # Kullanılan classifier train edildikten sonra test girdileri kullanılarak bir y skor matrisi elde edilir.
    # Bu y skor matrisi test verilerinin sırayla hastalık seviyesine ait olduğuna ait olasılık değerleri içermektedir.
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    y_score = clf.fit(X_train, Y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # fit işlemi sonucu elde edilen olasılık değerleri kullanılarak,
    # her bir seviye değeri için ROC eğrisi ve eğrinin altında kalan alanın hesaplanması
    for i in range(n_stages):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()

    for i in range(n_stages):
        plt.plot(fpr[i], tpr[i], label='ROC curve of ' + stage_names[i] + ' (AUC = %0.2f)' % roc_auc[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print("ROC Curve is completed")

# ROC eğrisinin Keras sinir ağı modeli kullanılarak çizdirilmesi
def plot_roc2(Y_test, predictions, stage_names):
    n_stages = len(stage_names)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Predict işlemi sonucu elde edilen tahmin değerleri kullanılarak,
    # her bir seviye değeri için ROC eğrisi ve eğrinin altında kalan alanın hesaplanması
    for i in range(n_stages):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()

    for i in range(n_stages):
        plt.plot(fpr[i], tpr[i], label='ROC curve of ' + stage_names[i] + ' (AUC = %0.2f)' % roc_auc[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print("ROC Curve is completed")

# Eğitim ve doğrulama hatalarının grafiğinin çizdirilmesi
def plot_train_and_val_loss(history, epoch):
    plt.figure()
    epochs = range(1, epoch + 1)

    # Modelin train işleminde her bir epoch içinde hesapladığı tüm eğitim ve doğrulama hatalarının eldesi
    loss = history.history['loss']
    valLoss = history.history['val_loss']

    plt.plot(epochs, loss, 'bo', label='Train Loss', color='red')
    plt.plot(epochs, valLoss, 'b', label='Validation Loss', color='blue')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.show()