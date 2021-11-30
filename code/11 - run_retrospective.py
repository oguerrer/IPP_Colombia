'''
Este script corre el modelo calibrado para el periodo histórico y guarda las 
asignaciones presupuestales inferidas para cada indicador instrumental.

'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed

home =  os.getcwd()[:-4]

os.chdir(home+'/code/')
import ppi

 




def run_ppi_parallel(I0, A, R, alpha, cc, rl, betas, scalar, B_sequence, budget_hash, max_theo):
    outputs = ppi.run_ppi(I0, A=A, R=R, alpha=alpha, cc=cc, rl=rl, max_theo=max_theo,
            betas=betas, get_gammas=True, scalar=scalar, B_sequence=B_sequence, budget_hash=budget_hash)
    tsI, tsC, tsF, tsP, tsD, tsS, times, H, gammas = outputs
    return (tsI, tsP)





# Dataset
df = pd.read_csv(home+"data/base_final.csv")
colYears = [col for col in df.columns if col.isnumeric()]

parallel_processes = 50
sample_size = 10000



A = np.loadtxt(home+'data/red/A.csv', dtype=float, delimiter=",")  
series = df[colYears].values
N = len(df)

# Initial factors
adf = pd.read_csv(home+"data/parametros.csv")
alphas = adf.alphas.values.copy()
max_steps = int(adf.steps.values[0])
betas = adf.beta.values.copy()
num_years = adf.years.values[0]
scalar = adf.scalar.values[0]
min_value = adf.min_value.values[0]

# Build variables
G = series[:,-1] - series[:,0]
G *= scalar
G[G<min_value] = min_value
G += series[:,0]*scalar
I0 = series[:,0]*scalar
num_years = series.shape[1]

R = df.Instrumental.values==1
R_idx = np.where(R)[0]
R_idx2new = dict(zip(R_idx, range(len(R_idx))))


dfb = pd.read_csv(home+"data/base_presupuesto.csv")
dfb = dfb.groupby('ODS').sum()
programs = sorted(dfb.index.values.astype(float).astype(int))

B_sequence = [[] for target in programs]
subperiods = max_steps/num_years
for i, program in enumerate(programs):
    for year in colYears:
        for x in range(int(subperiods)):
            B_sequence[i].append( dfb[dfb.index==int(float(program))][year].values[0]/subperiods )
    
B_sequence = np.array(B_sequence)



max_theo = df['Máx teórico'].values*scalar
cc = df['control_corrupcion'].values[0]
rl = df['estado_de_derecho'].values[0]

output_data = []


indis = df.Abreviatura.values
budget_hash = dict([ (i, ods[(ods!='nan') & (ods != '10.0')].astype(float).astype(int).tolist()) for i, ods in enumerate(df[['ODS1', 'ODS2']].values.astype(str)) ])




print('Runing model ...')
fin_vals = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)(I0, A, R, alphas, cc, rl, betas, scalar, B_sequence, budget_hash, max_theo) for itera in range(sample_size))

output_data_indi = []
output_data_P = []
for tsI, tsP in fin_vals:
    for i, serie in enumerate(tsI):
        output_data_indi.append([i] + serie.tolist())
        if R[i]:
            output_data_P.append([i] + tsP[R_idx2new[i]].tolist())



dfp = pd.DataFrame(output_data_P, columns=['indicador']+list(range(len(serie))))
dfp.groupby('indicador').mean().to_csv(home+"data/sims/retrospective_Ps.csv", index=False)













































