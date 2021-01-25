import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
df = pd.read_csv("DWH_Training.csv", sep=',',
                  names = ["index","Height", "weight", "gender"])
femaleSet=df.loc[df['gender'] == -1]
maleSet=df.loc[df['gender'] == 1]
dfTest = pd.read_csv("BIG_DWH_Training.csv", sep=',',
                  names = ["index","height", "weight", "gender","distance"])
# testSet = dfTest.iloc[:, 1:3].values
resultSet = dfTest.iloc[:, 3].values
def calculateCentroid(dataSet, labelx, labely):
    n = len(dataSet)
    centroid = np.array([np.sum(dataSet[labelx])/n, np.sum(dataSet[labely])/n])
    return centroid

def calculateLC(centM, centP):
    w = 2 * (centP - centM)
    b = np.square(np.sqrt(np.sum(np.square(centM)))) - np.square(np.sqrt(np.sum(np.square(centP))))
    return (w,b)

centPointM = calculateCentroid(maleSet,'Height', 'weight' )
centPointF = calculateCentroid(femaleSet, 'Height', 'weight')
init_w, init_b = calculateLC(centPointM, centPointF)


def SGD(data, b_val, w_val, B, C, T):
    for t in range(1,T):
        step = 1 / t
        # dataSet = df.sample(B).iloc[:, 1:4].values
        dataSet = getSampleDataSet(data, B)
        sum_der_b = 0
        sum_der_w = 0
        for i in range(len(dataSet)):
            x = np.array([[dataSet[i][0]], [dataSet[i][1]]])
            y = dataSet[i][2]
            checkVal = y * ((w_val.T.dot(x))[0] + b_val)
            if checkVal < 1:
                sum_der_b = sum_der_b - y
                sum_der_w = sum_der_w - ( y * x)

        w_val = w_val - (step * (w_val + C * sum_der_w))
        b_val = b_val - (step *(C * sum_der_b))
    return (b_val, w_val)


def getSampleDataSet(sampleData, val):
    tempList = list(sampleData)
    retList = list()
    while len(retList) < val & len(tempList)>0:
        index = random.randrange(len(tempList))
        retList.append(tempList.pop(index))
    return  retList


def callSGD(trainData, B_val, C_val):
    startTime = time.time()
    for i in range(1):
        trainSet = trainData
        (updB, updW) = SGD(trainSet, init_b, np.array([[init_w[0]], [init_w[1]]]), B_val, C_val, 10)
    endTime = time.time()
    print("Average Time taken = ", (endTime - startTime))
    print("Total Time taken =", (endTime - startTime) * 10)
    return (updB, updW)


(temp_b, temp_w) = callSGD(dfTest.iloc[:, 1:4].values, 50, 1)
plt.show()
