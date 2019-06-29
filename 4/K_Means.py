# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:12:58 2017

@author: student
"""

import numpy as np  
import matplotlib.pyplot as plt  


# 欧氏距离  
def euclDistance(vector1, vector2):  
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))
 
# 随机选取k个初始质心
def initCentroids(dataSet, k):  
    numSamples, dim = dataSet.shape   
    centroids = np.zeros((k, dim))         
    for i in range(k):  
        index = int(np.random.uniform(0, numSamples))  
        centroids[i, :] = dataSet[index, :]  
    return centroids  

# 聚类
def kmeans(dataSet, k):  
    numSamples = dataSet.shape[0]   
    clusterAssment = np.mat(np.zeros((numSamples, 2)))  
    clusterChanged = True  
    centroids = initCentroids(dataSet, k)  
    while clusterChanged:  
        clusterChanged = False  
        for i in range(numSamples):  
            minDist  = 100000.0  
            minIndex = 0  
            for j in range(k):  
                distance = euclDistance(centroids[j, :], dataSet[i, :])  
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j  
            if clusterAssment[i, 0] != minIndex:  
                clusterChanged = True  
                clusterAssment[i, :] = minIndex, minDist**2   
        for j in range(k):  
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)  
    return centroids, clusterAssment  


# 可视化
def showCluster(dataSet, k, centroids, clusterAssment):  
    numSamples, dim = dataSet.shape  
    if dim != 2:  
        print ("Sorry! The dimension must be 2! ")  
        return 1  

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):  
        print ("Sorry! k is too large! ")  
        return 1 
 
    for i in range(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  

    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  

    plt.title('K-means clustering')
    plt.xlabel('weight(kg)')
    plt.ylabel('height(cm)')
    plt.savefig('K-means.jpg')
    plt.show()