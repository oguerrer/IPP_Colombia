'''

Corre simulaciones contrafactuales usando los presupuestos optimizados.

'''

import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]




os.chdir(home+'/code/')
import ppi
from functions import *



parallel_processes = 20
sample_size = 1000


df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]



dft = df
df_params = pd.read_csv(home+"data/modeling/parameters.csv")



# Parameters
alphas = df_params.alpha.values.copy()
betas = df_params.beta.values.copy()
T = int(df_params['T'].values[0])
scalar = df_params.scalar.values[0]
min_value = df_params.min_value.values[0]
num_years = df_params.years.values[0]
sub_periods = int(T/num_years)
years_forward = 20

# Indicators
N = len(dft)
R = dft.Instrumental.values.copy()
n = R.sum()
I0 = df['2020'].values.copy()*scalar
Imax = np.ones(N)*scalar

# Network
A = np.loadtxt(home+"data/modeling/network.csv", delimiter=',')

# Governance
qm = np.ones(n)*dft.Monitoreo.values[0]
rl = np.ones(n)*dft.EstadoDeDerecho.values[0]


for interest in [0, 100, 1000]:

    print(interest)    

    # Budget
    df_expt = pd.read_csv(home+"data/modeling/optimal_budget_"+str(interest)+".csv")
    Bs_retro_tot = np.tile(df_expt.gasto.values, (years_forward,1)).T
    Bs = [[] for g in range(len(df_expt))]
    for i, year in enumerate(range(2021, 2020+years_forward+1)):
        for j in range(sub_periods):
            for index, row in df_expt.iterrows():
                Bs[index].append( Bs_retro_tot[index,i]/sub_periods )
    Bs = np.array(Bs)
    usdgs = df_expt.values[:,0]
    sdg2index = dict(zip(usdgs, range(len(usdgs))))
    sdgs = dft.MetaODS.values
    B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])
    
    
    # Simulation
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
                (I0, alphas, betas, A, R, qm, rl, Imax, Bs, B_dict, T, scalar) for itera in range(sample_size)))
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    aver_Is = np.mean(tsI, axis=0)
    aver_Ps = np.mean(tsP, axis=0)
    
    
    
    dfi = pd.DataFrame(np.hstack([[[c] for c in df.Abreviatura], aver_Is]), columns=['Abreviatura']+list(range(aver_Is.shape[1])))
    dfi.to_csv(home+"data/sims/optimal_prospective_"+str(interest)+".csv", index=False)
    
    dfp = pd.DataFrame(np.hstack([[[c] for c in df[df.Instrumental==1].Abreviatura ], aver_Ps]), columns=['Abreviatura']+list(range(aver_Ps.shape[1])))
    dfp.to_csv(home+"data/sims/optimal_prospective_Ps_"+str(interest)+".csv", index=False)
    
    
    
    

















