import numpy as np

"""
Contributions: Sushruth Badri
Comments: Erik Norén 14/12-22
"""

def findClassWeights(ds_train):
    '''
    Find weights depending on binary class distribution in training data.
    
    :param ds_train: training data set
    
    :return dictionary of weight per binary class
    '''
    total=len(ds_train.labels)
    
    weight_for_0 = (1 / (total - np.count_nonzero(ds_train.labels))) * (total / 2.0)
    weight_for_1 = (1 / np.count_nonzero(ds_train.labels)) * (total / 2.0)

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    return {0: weight_for_0, 1: weight_for_1}