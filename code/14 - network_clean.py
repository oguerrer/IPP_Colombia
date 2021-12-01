'''

Toma la red estimada y corrige los pesos de los enlaces así como signos que no 
son realistas.

'''


import matplotlib.pyplot as plt
import os, warnings, pycountry
import pandas as pd
import numpy as np


warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]



A = np.loadtxt(home+'/data/preprocessed/network_sparsebn.csv', dtype=float, delimiter=" ")  
links = A.flatten()
links = links[links!=0]
all_links = links


perb = np.percentile(all_links, 5)
pert = np.percentile(all_links, 95)






SDGs = df.ODS.values

A = np.loadtxt(home+'/data/preprocessed/network_sparsebn.csv', dtype=float, delimiter=" ")  

links = A.flatten()
links = links[links!=0]
A[A>pert] = 0
A[A<perb] = 0



for index, row in df.iterrows():

    indio = row.Abreviatura
                    
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if SDGs[i] == SDGs[j] and A[i,j] < 0:
            A[i,j] = 0
                                    

np.savetxt(home+"data/modeling/network.csv", A, delimiter=',')






























































































