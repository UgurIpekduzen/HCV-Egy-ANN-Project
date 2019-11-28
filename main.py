from backprop import NeuralNetwork
import numpy as np
import pandas as pd
import math

def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def parsePandasData(path):
    data = pd.read_csv(path)

    listCSVData = []
    df = pd.DataFrame(data, columns=data.columns.values)
    for _, value in df.iterrows():
        dataRow = []
        for i in range(data.columns.values.shape[0]):
            dataRow.append(value[i])
        listCSVData.append(dataRow)

    return listCSVData



if __name__ == "__main__":

    dataset = parsePandasData('./input/HCV-Egy-Data.csv')
    # dataset = np.random.uniform(low=-1, high=1, size=(1, 2))
    print("GİRDİLER: \n")
    print(dataset)

    n = 0.1
    print("\nÖĞRENME ORANI: " + str(n) + "\n")
    E = 0.2
    nn = NeuralNetwork(29, 29, 29)
    trained = nn.train(dataset, n, E)
