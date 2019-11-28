from activation import sigmoid
import numpy as np
from math import *
from decimal import *

class NeuralNetwork:
    def __init__(self, intInputNumber, intOutputNumber, intHiddenNeuronNumber=0, intLayerNumber=2):
        self.network = []

        if intHiddenNeuronNumber is 0:
            intHiddenNeuronNumber = intInputNumber

        # listHiddenLayerWeights = np.random.uniform(low=-1, high=1, size=(intInputNumber, intHiddenNeuronNumber))
        # self.network.append(listHiddenLayerWeights)
        #
        # listOutputLayerWeights = np.random.uniform(low=-1, high=1, size=(intHiddenNeuronNumber, intOutputNumber))
        # self.network.append(listOutputLayerWeights)

        listHiddenLayerWeights = np.ones((intInputNumber, intHiddenNeuronNumber))
        self.network.append(listHiddenLayerWeights)

        listOutputLayerWeights = np.ones((intHiddenNeuronNumber, intOutputNumber))
        self.network.append(listOutputLayerWeights)

        self.bias = np.random.uniform(low=0, high=1, size=intLayerNumber)
        print("\nBIAS DEĞERLERİ: ")
        print(self.bias)

        self.listTargets = np.random.randint(2, size=intOutputNumber)
        print("\nHEDEF ÇIKTILAR: ")
        print(self.listTargets)

    def activate(self, weights, inputs):
        net = 0.0
        for i in range(len(weights)):
            net += np.multiply(weights[i], inputs[i])
        return net

    def forwardPropagation(self, row, networkWeights, count = 0):
        listInputs = row
        intBiasCount = 0
        for layer in networkWeights:
            listHidLayerInputs = []
            for neuron in layer:
                activation = self.activate(neuron, listInputs) + self.bias[intBiasCount]
                output = sigmoid(activation)
                listHidLayerInputs.append(output)
            listInputs = listHidLayerInputs
            intBiasCount += 1

        return listInputs

    def backwardPropagationError(self, outputs, networkWeights):
        intOutputNumber = len(outputs)
        listAllErrorsBackward = []
        for i in reversed(range(len(networkWeights))):
            layer = networkWeights[i]
            listLayerErrors = []
            for j in range(intOutputNumber):
                listNeuronErrors = []
                for k in range(layer.shape[1]):
                    delta_0 = (-2/intOutputNumber) * (self.listTargets[j] - outputs[j])
                    delta_1 = outputs[j] * (1 - outputs[j])
                    delta_2 = layer[j][k]
                    delta_weight = delta_0 * delta_1 * delta_2

                    listNeuronErrors.append(np.array(delta_weight))

                listLayerErrors.append(np.array(listNeuronErrors))

            listAllErrorsBackward.append(np.array(listLayerErrors))

        listAllErrorsBackward = np.array(listAllErrorsBackward)

        print("AĞIRLIK GERİ YAYILIM HATALARI: \n")
        print(np.flip(listAllErrorsBackward, 0))
        return np.flip(listAllErrorsBackward, 0)

    def updateWeights(self, errors, floatLearningRate, networkWeights):
        allNetworkWeights = []
        print("\nNETWORK'DEKİ MEVCUT AĞIRLIKLAR: \n")
        print(networkWeights)
        for i in reversed(range(len(networkWeights))):
            layerWeights = networkWeights[i]
            layerErrors = errors[i]
            allNetworkWeights.append(np.subtract(layerWeights, floatLearningRate * layerErrors))
        print("\nNETWORK'DEKİ GÜNCEL AĞIRLIKLAR: \n")
        print(allNetworkWeights)
        return allNetworkWeights

    def truncate(self, number, digits):
        stepper = 10.0 ** digits
        return trunc(stepper * number) / stepper

    def areErrorsEqual(self, SSE, floatMinErrorRate):
        # floatMinErrorRate = 0.0
        decimalPlaces = abs(Decimal(str(floatMinErrorRate)).as_tuple().exponent)
        truncatedSSE = self.truncate(SSE, decimalPlaces)

        return True if truncatedSSE == floatMinErrorRate else False

    def train(self, listDataSet, floatLearningRate, floatMinErrorRate):

        trainedNetworks = []

        for row in listDataSet:
            # intCount = 0  # Layer'lara ait bias değerlerini çağırır

            # Ağa giriş yapan değerler için ham ağırlık değerlerini kullanır
            defaultWeights = self.network

            intEpoch = 0

            while True:
                print("\nNESİL:" + str(intEpoch) + "\n")
                # outputs = self.forwardPropagation(row, intCount)

                outputs = self.forwardPropagation(row, defaultWeights)
                print("\nÇIKTILAR: \n")
                print(outputs)

                # Sum Square Error
                SSE = np.sum(np.square(np.square(np.subtract(self.listTargets, outputs))))
                print("\nTOPLAM HATA: " + str(SSE) + "\n")

                errors = self.backwardPropagationError(outputs, defaultWeights)

                # Giriş katmanındaki değerlere göre ağdaki tüm ağırlıkları günceller
                defaultWeights = self.updateWeights(errors, floatLearningRate, defaultWeights)

                isEqual = self.areErrorsEqual(SSE, floatMinErrorRate)
                if self.areErrorsEqual(SSE, floatMinErrorRate):
                    break
                else:
                    intEpoch += 1
                # intCount += 1
            trainedNetworks.append(defaultWeights)

        return trainedNetworks


