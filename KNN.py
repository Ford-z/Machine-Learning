from numpy import *
import numpy as np
import operator
import pandas as pd
from scipy.spatial import distance  # compute distances


# create dataset and classes
def createDataSet():
    #group = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    #label = ['A', 'A', 'B', 'B']
    df = pd.DataFrame(pd.read_csv('./traindata.csv', header = 0,usecols=['x1','x2']))
    group=df.values
    dt = pd.DataFrame(pd.read_csv('./traindata.csv', header=0))
    label=np.array(dt['class'])
    label=label.tolist()
    #print(df)

    return group, label


# function KNN
def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    diff = tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  ###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5

    ##对距离进行排序
    sortedDistIndex = argsort(dist)  ##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes
