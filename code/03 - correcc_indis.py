'''
Este script toma los indicadores imputados, y corrige las imputaciones en caso de que
estas presenten valores inválidos o volatilidad mayor a la de los datos empíricos.

'''

import matplotlib.pyplot as plt
import numpy as np
import os, copy
import pandas as pd

home =  os.getcwd()[:-4]





dfn = pd.read_csv(home+"data/base_norm.csv")
dfi = pd.read_csv(home+"data/base_imput.csv")
colYears = [year for year in dfn.columns if str(year).isnumeric()]
colOthers = [year for year in dfn.columns if not str(year).isnumeric()]
years = np.array([int(c) for c in dfn.columns if c.isnumeric()])



new_rows = []

lower = 0
upper = 0
i = 0
for index, row in dfi.iterrows():

    vals = row[colYears].values.astype(float)
    valsi = dfn.loc[index, colYears].values.astype(float)
    
    first = np.where(~np.isnan(valsi))[0][0]
    last = np.where(~np.isnan(valsi))[0][-1]
    
    vv = vals[first:last+1]
    vv = np.abs(vv[1::] - vv[0:-1])
    
    vvf = vals[0:first+1]
    vvf = np.abs(vvf[1::] - vvf[0:-1])
    
    vvl = vals[last::]
    vvl = np.abs(vvl[1::] - vvl[0:-1])
    
    ulimit = row['Máx teórico']+.001
    llimit = row['Min teórico']-.001
    
    if (len(vvf) > 0 and np.max(vvf) > np.max(vv)) or (np.min(vals[0:first+1])<llimit) or (np.max(vals[0:first+1])>ulimit):
        
        lower += 1
        
        ref_val = vals[first]
        while (np.max(vvf) > np.max(vv)) or (np.min(vals[0:first+1])<llimit) or (np.max(vals[0:first+1])>ulimit):
            
            diff = 0.999*(vals - ref_val)
            vals[0:first] = ref_val + diff[0:first]
            vvf = vals[0:first+1]
            vvf = np.abs(vvf[1::] - vvf[0:-1])


    if (len(vvl) > 0 and np.max(vvl) > np.max(vv)) or (np.min(vals[last::])<llimit) or (np.max(vals[last::])>ulimit):
        fig = plt.figure( figsize=(6,3.5) )

        upper += 1        
        
        if np.sum(vv==0) == len(vv):
            vals[last+1::] = vals[last]

        else:
            plt.plot(years.astype(int), vals, linewidth=2)
            # plt.plot(years.astype(int), valsi, linewidth=2)
            
            ref_val = vals[last]
            while (np.max(vvl) > np.max(vv))  or (np.min(vals[last::])<llimit) or (np.max(vals[last::])>ulimit):
                
                diff = 0.999*(vals - ref_val)
                vals[last+1::] = ref_val + diff[last+1::]
                vvl = vals[last::]
                vvl = np.abs(vvl[1::] - vvl[0:-1])
                
            plt.plot(years[last::].astype(int), vals[last::], linewidth=2)
            plt.plot(years[0:last+1].astype(int), vals[0:last+1], linewidth=2)
            plt.xticks(years.astype(int)[0::2])
            plt.xlabel('year', fontsize=14)
            plt.ylabel('value', fontsize=14)
            plt.legend(['uncorrected extrapolation', 'corrected extrapolation', 'empirical data'])
            plt.tight_layout()
            # plt.savefig(path+'example_correcition.pdf')
            plt.show()
        
        
    new_rows.append(vals.tolist() + row[colOthers].values.tolist())
    i+=1


print(lower, upper)


dff = pd.DataFrame(new_rows, columns=colYears+colOthers)   
dff.to_csv(home+'data/base_correc.csv', index=False)






































































