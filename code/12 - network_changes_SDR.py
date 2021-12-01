'''

Calcula las primeras diferencias en as series de tiempo de los indicadores y
prepara un achivo para estimar la red de interdependencias con un script the R.

'''



import matplotlib.pyplot as plt
import os, warnings, pycountry
import pandas as pd
import numpy as np


warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]



M = df[colYears].values
changes = M[:,1::] - M[:,0:-1]
np.savetxt(home+"data/preprocessed/network_changes.csv", changes, delimiter=',')


























































