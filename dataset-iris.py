import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt


def parseRowsAndColumns():
    rawData = np.loadtxt('./input/iris.csv', delimiter=",")
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
        patient['Sepal Length'] = int(dataColumns[i][0])
        patient['Sepal Width'] = int(dataColumns[i][1])
        patient['Petal Length'] = int(dataColumns[i][2])
        patient['Petal Width'] = int(dataColumns[i][3])

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


dataset = setDataFrame()
plot_corr(dataset)
