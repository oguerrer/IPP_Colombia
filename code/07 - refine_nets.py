'''
Este script toma la red estimada y corrige los pesos en caso de haber valores
aberrantes o falsos positivos (enlaces negativos al interior de un mismo ODS).

'''

import matplotlib.pyplot as plt
import numpy as np
import os, copy
import pandas as pd


home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/base_final.csv")
colYears = [c for c in df.columns if c.isnumeric()]


A = np.loadtxt(home+'/data/red/sparsebn.csv', dtype=float, delimiter=" ")  

SDGs = dict(zip(range(A.shape[0]), [set(pair[~np.isnan(pair)]) for pair in df[['ODS1', 'ODS2']].values]))
                    
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if len(SDGs[i].intersection(SDGs[j])) > 0 and A[i,j] < 0:
            A[i,j] = 0
                                    

W = A.flatten()
W = W[W!=0]
W[W<-1] = 0
W[W>1] = 0
A[A!=0] = W

np.savetxt(home+"data/red/A.csv", A, delimiter=',')





























































