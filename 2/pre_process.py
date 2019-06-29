# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:09:07 2018

@author: student
"""
import re
import numpy as np
import os

def extra_data(name):
    data_ori = open(name).readlines()
    data_pro = []
    for data in data_ori:
        data_pro.append(re.split('\t| ', data))

    flavour_index = []
    date_ori = []
    for data in data_pro:
        flavour_index.append(re.sub("\D", "", data[1]))
        date_ori.append(re.sub("\D", "", data[2]))

    flavour_index = [int(x) -1 for x in flavour_index]
    date_ori = [int(x) for x in date_ori]

    data = np.vstack((date_ori, flavour_index)).T

    index = []
    for i in range(len(flavour_index)):   
        if data[i][1] > 15:
            index.append(i)
  
    data = np.delete(data, index, 0)

    def dense_to_one_hot(labels_dense, num_classes):  
 
        num_labels = labels_dense.shape[0]  
        index_offset = np.arange(num_labels) * num_classes  
        labels_one_hot = np.zeros((num_labels, num_classes))  
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  
        return labels_one_hot  


    flavour_temp = data[:,1]
    flavour = dense_to_one_hot(flavour_temp, 15)

    date_tmp = data[:, 0]

    days = []
    for i in list(set(date_tmp)):
        days.append(list(date_tmp).count(i))

    def day_sum(date_tmp, flavour, day):
        res = []
        for i in range(len(date_tmp) - 1):
            if i < day:
                res.append(flavour[i])
        return sum(res)

    res = []
    for i in days:
        res.append(day_sum(date_tmp, flavour, i))
    return np.array(res)
    
data = []
data_list = os.listdir('train data')
for name in data_list:
    data.append(extra_data('train data/' + name))

data = np.vstack((data[0], data[1], data[2], data[3], data[4], data[5], data[6]))




        




    

