# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:32:15 2017

@author: student
"""

import numpy as np

def decodechrom(pop_t, data_f, chrom_length):
    temp = []
    for i in range(chrom_length):
        if pop_t[i] == 1:
            temp.append(data_f[i, :])
    return temp

def calfeatures(pop, data, chrom_length):
    features = []
    pop_size = len(pop)
    for i in range(pop_size):
        temp = decodechrom(pop[i], data[1:, :], chrom_length)
        features.append(np.array(temp))
    return np.array(features)


def calFeatures_sex(pop, data, chrom_length):
    features_men = []
    features_women = []
    pop_size = len(pop)
    features = calfeatures(pop, data, chrom_length)
    for i in range(pop_size):
        temp_men = []
        temp_women = []
        for index, value in enumerate(data[0, :]):
            if value == 1:
                temp_men.append(features[i][:, index])
            else:
                temp_women.append(features[i][:, index])
        features_men.append(np.array(temp_men).T)
        features_women.append(np.array(temp_women).T)
    return features_men, features_women
