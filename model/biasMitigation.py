import numpy as np

def findClassWeights(ds_train):
    total=len(ds_train.labels)
    
    weight_for_0 = (1 / (total - np.count_nonzero(ds_train.labels))) * (total / 2.0)
    weight_for_1 = (1 / np.count_nonzero(ds_train.labels)) * (total / 2.0)

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    return {0: weight_for_0, 1: weight_for_1}