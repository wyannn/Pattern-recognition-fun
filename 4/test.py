# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 11:14:20 2017

@author: student
"""
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from itertools import cycle  
import xlrd as xd
import K_Means
import numpy as np

# K-means聚类算法

# 加载数据 
print('K-means clustering')
data = xd.open_workbook('data.xls')
table = data.sheets()[0]
height = table.col_values(3)[1:]
weight = table.col_values(4)[1:]
dataSet = np.vstack((height, weight)).T 
# 聚类
k = 2  
centroids, clusterAssment = K_Means.kmeans(dataSet, k)
# 可视化
K_Means.showCluster(dataSet, k, centroids, clusterAssment)

# 分层聚类算法

#设置分层聚类函数
print('hierarchical clustering')
linkages = ['ward', 'average', 'complete']
n_clusters_ = 2
ac = AgglomerativeClustering(linkage=linkages[2],n_clusters = n_clusters_)
# 训练数据
ac.fit(dataSet)
# 每个数据的分类
lables = ac.labels_
# 可视化
plt.figure(1)
plt.clf()
colors = cycle(' ')

for k, col in zip(range(n_clusters_), colors):
    my_members = lables == k
    plt.plot(dataSet[my_members, 0], dataSet[my_members, 1], col + 'o')
    
plt.title('hierarchical clustering')
plt.xlabel('height(cm)')
plt.ylabel('weight(kg)')
plt.savefig('hierarchical_clustering.jpg')
plt.show()
