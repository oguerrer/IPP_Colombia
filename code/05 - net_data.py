'''
Este script obtiene las primeras diferencias intertemporales de los indicadores.
Los datos resultantes son utilizados para estimar la red de interdependencias
entre indicadores que será uno de los insumos de IPP.

'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/base_norm.csv")
colYears = [c for c in df.columns if c.isnumeric()]


M = df[colYears].values


maxes = []
for a in np.linspace(0,1,15):
    y = [np.sum(~np.isnan(M[:,i::]))/(M[:,i::].size) + a*(M[:,i::].shape[1]/M.shape[1]) for i in range(M.shape[1])]
    maxes.append(np.argmax(y))
    
    plt.plot(y)
    plt.plot(np.argmax(y), np.max(y), '.k')


maxes = np.array(maxes)
diff = maxes[0:-1] - maxes[1::]



dff = pd.read_csv(home+"data/base_correc.csv")

newColYears = colYears[14::]
new_rows = dff[newColYears].values.copy()
new_rows = np.diff(new_rows, axis=1)
            
for i, change in enumerate(new_rows):
    if np.sum(change) == 0:
        new_rows[i] = np.random.rand(len(change))*10e-18
        
        
np.savetxt(home+"data/red/changes.csv", new_rows, delimiter=',')









































































































