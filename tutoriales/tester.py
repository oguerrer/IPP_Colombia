'''

Calibra los parámetros del modelo usando los datos históricos.

'''


import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

import requests
url = 'https://raw.githubusercontent.com/oguerrer/IPP_Colombia/main/code/ppi.py'
r = requests.get(url)
with open('ppi.py', 'w') as f:
    f.write(r.text)
import ppi

url = 'https://raw.githubusercontent.com/oguerrer/IPP_Colombia/main/tutoriales/functions.py'
r = requests.get(url)
with open('functions.py', 'w') as f:
    f.write(r.text)
from functions import *

url = 'https://raw.githubusercontent.com/oguerrer/IPP_Colombia/main/tutoriales/calibrator.py'
r = requests.get(url)
with open('calibrator.py', 'w') as f:
    f.write(r.text)
import calibrator



df = pd.read_csv('https://raw.githubusercontent.com/oguerrer/IPP_Colombia/main/data/modeling/indicators_sample_final.csv')
colYears = [col for col in df.columns if str(col).isnumeric()]





parallel_processes = 10


num_years = len(colYears)
scalar = 100
min_value = 1e-2

sub_periods = 4
T = len(colYears)*sub_periods




# Extract country data
dft = df
df_expt = pd.read_csv('https://raw.githubusercontent.com/oguerrer/IPP_Colombia/main/data/modeling/budget_target.csv')

# Indicators
series = dft[colYears].values
N = len(dft)
Imax = np.ones(N)*scalar
R = dft.Instrumental.values.copy()
n = R.sum()
I0, IF = get_conditions(series, scalar, min_value, Imax)
success_emp = get_success_rates(series)


# Budget
Bs = np.tile(df_expt[colYears].mean(axis=1).values, (len(colYears),1)).T
usdgs = df_expt.values[:,0]
sdg2index = dict(zip(usdgs, range(len(usdgs))))
sdgs = dft.MetaODS.values
B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])
Bs = get_dirsbursement_schedule(Bs, B_dict, T)

# Network
A = np.loadtxt('https://raw.githubusercontent.com/oguerrer/IPP_Colombia/main/data/modeling/network.csv', dtype=float, delimiter=',')

# Governance
monitoreo = np.ones(n)*dft.Monitoreo.values[0]
estadoDeDerecho = np.ones(n)*dft.EstadoDeDerecho.values[0]



parallel_processes = 2 # número de procesos paralelos
tolerance = .1 # tolerancia del error promedio
parametros_calibrados = calibrator.calibrate(I0=I0, A=A, R=R, qm=monitoreo, rl=estadoDeDerecho,  
    Bs=Bs, B_dict=B_dict, T=T, scalar=scalar, IF=IF, success_emp=success_emp, num_years=num_years,
    min_value=min_value, tolerance=.9, parallel_processes=2)





