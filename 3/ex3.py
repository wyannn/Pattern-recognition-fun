# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:51:54 2017

@author: student
"""

import xlrd as xd
import numpy as np

from sklearn.metrics import roc_auc_score
from geneEncoding import geneEncoding
from feature_normalize import feature_normalize
from evaluate import evaluate
from sklearn import svm
from selection import selection
from crossover import crossover
from mutation import mutation
import matplotlib.pyplot as plt
from calfitValue import calfitValue
from calFeatures_sex import calFeatures_sex
from sklearn.decomposition import PCA  

#读取数据（训练集）
data = xd.open_workbook('data.xls')
table = data.sheets()[0]
sex = table.col_values(1)[1:]
height = table.col_values(3)[1:]
weight = table.col_values(4)[1:]
likemath = table.col_values(6)[1:]
likeart = table.col_values(7)[1:]
likesport = table.col_values(8)[1:]
likepattern = table.col_values(9)[1:]
data = np.vstack((sex, height, weight, likemath, likeart, likesport, likepattern))

#遗传算法
pop_size = 63	                          
generation_num = 100                    
chrom_length = 6		                      
pc = 0.6			                             
pm = 0.01             
results = []		      
prior_men = np.sum(sex) / len(sex)
prior_women = 1.0 - prior_men 

pop = geneEncoding(pop_size, chrom_length)

for i in range(generation_num):
    features_men, features_women = calFeatures_sex(pop, data, chrom_length)
    fitness = calfitValue(features_men, features_women, prior_men, prior_women)        
    best_fit = max(fitness)
    best_individual = pop[fitness.index(best_fit)]
    results.append([best_fit, best_individual])
    selection(pop, fitness)	
    crossover(pop, pc)	
    mutation(pop, pm)
      
results.sort()

#特征出现的概率
height_probability = 0
weight_probability = 0
likemath_probability = 0
likeart_probability = 0
likesports_probability = 0
likepattern_probability = 0
for i in range(100):
    if results[i][1][0] == 1:
        height_probability += 1
    if results[i][1][1] == 1:
        weight_probability += 1
    if results[i][1][2] == 1:
        likemath_probability += 1
    if results[i][1][3] == 1:
        likeart_probability += 1
    if results[i][1][4] == 1:
        likesports_probability += 1
    if results[i][1][5] == 1:
        likepattern_probability += 1
        
#画图
X_plt = []
Y_plt = []
for i in range(generation_num):
    X_plt.append(i)
    Y_plt.append(results[i][0])

plt.plot(X_plt, Y_plt)
plt.show()
plt.savefig('iteration.jpg')

#数据处理  
best_feature = []
for i in range(chrom_length):
    if results[-1][1][i] == 1:
        best_feature.append(data[i, :])
      
best_feature = np.array(best_feature)
X_ga = feature_normalize(best_feature[:, :int(len(sex)*0.8)])
X_test_ga = feature_normalize(best_feature[:, int(len(sex)*0.8):])

Y = np.array(sex[:int(len(sex)*0.8)])
Y_test = np.array(sex[int(len(sex)*0.8):])

#SVM训练
clf = svm.SVC(gamma = 10)    
clf.fit(X_ga.T, Y)
predictions = clf.predict(X_test_ga.T)
SE2, SP2, ACC2 = evaluate(predictions, Y_test)
AUC2 = roc_auc_score(Y_test.T, predictions.T)
print('\n' + '-' * 35)
print('遗传算法选择的特征: %s' % results[-1][1])
print('SVM分类结果评估:')
print('SE: %s' % SE2)
print('SP: %s' % SP2)
print('ACC: %s' % ACC2)
print('AUC: %s' % AUC2)
print('身高特征出现的概率：%s%%' % height_probability)
print('体重特征出现的概率：%s%%' % weight_probability)
print('是否喜欢数学特征出现的概率：%s%%' % likemath_probability)
print('是否喜欢文学特征出现的概率：%s%%' % likeart_probability)
print('是否喜欢体育特征出现的概率：%s%%' % likesports_probability)
print('是否喜欢模式识别特征出现的概率：%s%%' % likepattern_probability)
print('-'*35 + '\n') 


#pca
pca = PCA(n_components = 2)#维数选择  
new_data = pca.fit_transform(data[1:, :].T)

X_pca = feature_normalize(new_data.T[:, :int(len(sex)*0.8)])
X_test_pca = feature_normalize(new_data.T[:, int(len(sex)*0.8):])

Y = np.array(sex[:int(len(sex)*0.8)])
Y_test = np.array(sex[int(len(sex)*0.8):])

#SVM训练
clf = svm.SVC(gamma = 10)    
clf.fit(X_pca.T, Y)
predictions = clf.predict(X_test_pca.T)
SE2, SP2, ACC2 = evaluate(predictions, Y_test)
AUC2 = roc_auc_score(Y_test.T, predictions.T)
print('\n' + '-' * 35)
print('pca算法选择的特征数目: %s' % sum(results[-1][1]))
print('pca算法选择特征的方差: %s' % pca.explained_variance_)
print('pca算法选择特征占总特征的百分比: %s' % pca.explained_variance_ratio_ )
print('SVM分类结果评估:')
print('SE: %s' % SE2)
print('SP: %s' % SP2)
print('ACC: %s' % ACC2)
print('AUC: %s' % AUC2)
print('-'*35 + '\n')  
