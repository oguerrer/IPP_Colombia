import matplotlib.pyplot as plt
import os, warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

home =  os.getcwd()

import ppi



## LOAD ALL THE BENCHMARKING DATA
df_param = pd.read_excel(home+'/template_indicators.xlsx', sheet_name='template')
I0 = df_param['initial_value'].values
IF = df_param['final_value'].values
R = df_param['instrumental'].values
qm = df_param['monitoring'].values
rl = df_param['rule_of_law'].values
Imin = df_param['min_value'].values
Imax = df_param['max_value'].values
success_rates = df_param['success_rate'].values
alpha = df_param['alpha'].values
alpha_prime = df_param['alpha_prime'].values
beta = df_param['beta'].values
IF[I0==IF] = IF[I0==IF] * 1.0001

df_B = pd.read_excel(home+'/template_budget.xlsx', sheet_name='template_expenditure')
Bs = df_B.expenditure
B_ids = dict(zip(df_B.program_ID, range(len(df_B))))

df_B = pd.read_excel(home+'/template_budget.xlsx', sheet_name='template_relation_table').dropna(axis=1, how='all')
B_dict = {}
for index, row in df_B.iterrows():
    B_dict[int(row[0]-1)] = [B_ids[c] for c in row[df_B.columns[1::]].values if not np.isnan(c)]
    

df_A = pd.read_excel(home+'/template_network.xlsx', sheet_name='template_network')
A = df_A.values[:,1::].astype(float)



## CALIBRATION WITH IMPERFECT BUDGET MATRIX AND INDICATOR PROPERTIES
parameters = ppi.calibrate(I0, IF, success_rates, parallel_processes=4, Bs=Bs, 
                           B_dict=B_dict, A=A, R=R, qm=qm, rl=rl, verbose=True,
                           threshold=0.85, T=252)
dfp = pd.DataFrame(parameters[1::,:], columns=parameters[0])
dfp.to_excel(home+'/app_outputs.xlsx')




































