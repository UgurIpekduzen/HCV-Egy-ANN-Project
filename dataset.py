import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

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
        patient['Nausea / Vomting'] = int(dataColumns[i][4])
        patient['Headache'] = int(dataColumns[i][5])
        patient['Diarrhea'] = int(dataColumns[i][6])
        patient['Fatigue'] = int(dataColumns[i][7])
        patient['Bone ache'] = int(dataColumns[i][8])
        patient['Jaundice'] = int(dataColumns[i][9])
        patient['Epigastria Pain'] = int(dataColumns[i][10])
        patient['WBC'] = int(dataColumns[i][11])
        patient['RBC'] = int(dataColumns[i][12])
        patient['HGB'] = int(dataColumns[i][13])
        patient['Plat'] = int(dataColumns[i][14])
        patient['AST1'] = int(dataColumns[i][15])
        patient['ALT1'] = int(dataColumns[i][16])
        patient['ALT4'] = int(dataColumns[i][17])
        patient['ALT12'] = int(dataColumns[i][18])
        patient['ALT24'] = int(dataColumns[i][19])
        patient['ALT36'] = int(dataColumns[i][20])
        patient['ALT48'] = int(dataColumns[i][21])
        patient['RNA Base'] = int(dataColumns[i][22])
        patient['RNA4'] = int(dataColumns[i][23])
        patient['RNA12'] = int(dataColumns[i][24])
        patient['RNA EOT'] = int(dataColumns[i][25])
        patient['RNA EF'] = int(dataColumns[i][26])
        patient['Baseline Histological Grading'] = int(dataColumns[i][27])
        patient['Baseline Histological Staging'] = int(dataColumns[i][28])
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




