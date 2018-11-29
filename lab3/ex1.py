import pandas as pd
from sub.min import *
import numpy as np

#link = https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease
if __name__ == "__main__":
    column = np.arange(25)
    feature = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod',
               'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
    x = pd.read_csv("ckd/chronic_kidney_disease.arff",
                    sep=',', na_values=['?', '\t?'], skiprows=0, header=None, usecols=column)
    x.info()

