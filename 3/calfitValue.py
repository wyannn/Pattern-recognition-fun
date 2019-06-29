# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:37:51 2017

@author: student
"""
import numpy as np


def calfitValue(features_men, features_women, prior_men, prior_women):
    fitness = []
    pop_size = len(features_men)
    for i in range(pop_size):
        Sw = prior_men * np.cov(features_men[i]) + prior_women * np.cov(features_women[i])
        sub1 = features_men[i] - np.mean(features_men[i], axis=1).reshape(-1,1)
        sub2 = features_women[i] - np.mean(features_women[i], axis=1).reshape(-1,1)
        Sb = prior_men * np.dot(sub1, sub1.T) + prior_women * np.dot(sub2, sub2.T)
        if (features_men[i].shape[0] == 6) or (features_men[i].shape[0] == 1):
            fit = (np.sum(Sb) - np.sum(Sw)) / 500 - 1 + features_men[i].shape[0] / 6
            fitness.append(fit)
        else:
            fit = (np.sum(Sb) - np.sum(Sw)) / 500 + features_men[i].shape[0] / 6
            fitness.append(fit)
    return fitness


