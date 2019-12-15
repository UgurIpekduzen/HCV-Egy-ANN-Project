import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dataset import *
from sklearn.metrics import plot_confusion_matrix
import itertools

def parseRowsAndColumns():
    rawData = np.loadtxt('./input/HCV-Egy-Data.csv', delimiter=",")
    splittedData = []
    for i in range(rawData.shape[0]):
        column = []
        for j in range(rawData.shape[1]):
            column.append(rawData[i][j])
        splittedData.append(column)
    return splittedData

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
        patient['Baseline_Histological_Grading'] = int(dataColumns[i][27])
        patient['Baseline_Histological_Staging'] = int(dataColumns[i][28])
        patientDict['data'].append(patient)

    patientDataFrame = DataFrame.from_dict(patientDict['data'])

    return patientDataFrame

def plot_corr(df):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df.corr(), cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Korelasyon Matrisi')
    plt.show()
    print("Çizildi")

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

def plot_cnf_matrix(target, predicted, classes, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title = ""
    titleOptions = ["Confusion Matrix with Normalization", "Confusion Matrix without Normalization"]
    plt.figure()
    cnfMatrix = confusion_matrix(y_true=target, y_pred=predicted, labels=classes)
    np.set_printoptions(precision=2)
    cmap = plt.cm.Blues


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
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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

def findRepeatedElements(x):
    _size = len(x)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if x[i] == x[j] and x[i] not in repeated:
                repeated.append(x[i])
    return repeated



