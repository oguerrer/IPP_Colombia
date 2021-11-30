'''
Este script hace el muestreo final de los indicadores y genera la base de datos
que será usada con IPP.

'''


import matplotlib.pyplot as plt
import numpy as np
import os, copy
import pandas as pd

home =  os.getcwd()[:-4]





df = pd.read_csv(home+"data/base_correc.csv")
colYears = [year for year in df.columns if str(year).isnumeric()]
colOthers = [year for year in df.columns if not str(year).isnumeric()]




dff = df[colYears[17::]+colOthers]
dff['Meta'] = dff['Meta MML'].values
dff.loc[dff.Meta.isnull(), 'Meta'] = dff['Meta PNUD'][dff.Meta.isnull()]
dff.to_csv(home+'data/base_final.csv', index=False)












