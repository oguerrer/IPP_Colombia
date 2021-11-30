'''
Este script toma los indicadores normalizados e imputa sus valores faltantes
mediante un modelo estadístico conocido como proceso de Gauss.

'''

import matplotlib.pyplot as plt
import numpy as np
import os, copy
import pandas as pd
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

home =  os.getcwd()[:-4]





df = pd.read_csv(home+"data/base_norm.csv")
colYears = [year for year in df.columns if str(year).isnumeric()]
colOthers = [year for year in df.columns if not str(year).isnumeric()]



new_rows = []
years = 1+np.arange(len(colYears))
for index, row in df.iterrows():
    
    vals = row[colYears].values.astype(float)

    if np.sum(np.isnan(vals)) > 0:
        
        bools = ~np.isnan(vals)
        occur = np.where(bools)[0]
        
        x = years[bools]
        y = vals[bools]
        X = x.reshape(-1, 1)
        
        kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)

        x_pred = years[~bools].reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        
        vals[np.in1d(years, x_pred.flatten())] = y_pred
        
        # plt.plot(vals)
        # plt.show()

        new_rows.append(vals.tolist() + row[colOthers].values.tolist())


dff = pd.DataFrame(new_rows, columns=colYears+colOthers)   
dff.to_csv(home+'data/base_imput.csv', index=False)











































































