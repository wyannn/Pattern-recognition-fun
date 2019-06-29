import random as rm
import math as mh

def randomized_data(sex):

    sex_men = []
    sex_women = []
    
    for index, value in enumerate(sex):
        if value == 1:
            sex_men.append(int(index))
        else:
            sex_women.append(int(index))
        
    index_men = rm.sample(sex_men, mh.ceil(len(sex_men) * 0.2))
    index_women = rm.sample(sex_women, int(len(sex_women) * 0.2))
    index = index_men + index_women
    
    return index


