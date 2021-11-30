'''
Este script realiza la calibración del modelo y genera un archivo con los parámetros.

'''


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed


home =  os.getcwd()[:-4]

os.chdir(home+'/code/')
import ppi





def run_ppi_parallel(I0, A, R, alpha, cc, rl, betas, scalar, B_sequence, budget_hash):
    outputs = ppi.run_ppi(I0, A=A, R=R, alpha=alpha, cc=cc, rl=rl,
            betas=betas, get_gammas=True, scalar=scalar, B_sequence=B_sequence, budget_hash=budget_hash)
    tsI, tsC, tsF, tsP, tsD, tsS, times, H, gammas = outputs
    return (tsI[:,-1], gammas)



def fobj2(I0, A, R, alpha, cc, rl, betas, scalar, sample_size, G, success_emp, B_sequence, budget_hash):
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
            (I0, A, R, alpha, cc, rl, betas, scalar, B_sequence, budget_hash) for itera in range(sample_size)))
    FIs = []
    gammas = []
    for sol in sols:
        FIs.append( sol[0] )
        for gamma in sol[1]:
            gammas.append( gamma )

    mean_indis = np.mean(FIs, axis=0)
    error_alpha = G - mean_indis
    mean_gamma = np.mean(gammas, axis=0)
    error_beta = success_emp - mean_gamma

    return error_alpha.tolist() + error_beta.tolist()






ith = 1
parallel_processes = 30
max_steps = 54

 
# Dataset
df = pd.read_csv(home+"data/base_final.csv")
colYears = np.array([c for c in df.columns if c.isnumeric()])

scalar = 100
min_value = 1e-2
N = len(df)

indis = df.Abreviatura.values
budget_hash = dict([ (i, ods[(ods!='nan') & (ods != '10.0')].astype(float).astype(int).tolist()) for i, ods in enumerate(df[['ODS1', 'ODS2']].values.astype(str)) ])

A = np.loadtxt(home+"data/red/A.csv", dtype=float, delimiter=",")  
series = df[colYears].values


# Build variables
G = series[:,-1] - series[:,0]
G *= scalar
G[G<min_value] = min_value
G += series[:,0]*scalar
I0 = series[:,0]*scalar
num_years = series.shape[1]

R = df.Instrumental.values



dfb = pd.read_csv(home+"data/base_presupuesto.csv")
dfb = dfb.groupby('ODS').sum()
programs = sorted(dfb.index.values.astype(float).astype(int))

B_sequence = [[] for target in programs]
subperiods = max_steps/len(colYears)
for i, program in enumerate(programs):
    for year in colYears:
        for x in range(int(subperiods)):
            B_sequence[i].append( dfb[dfb.index==int(float(program))][year].values[0]/subperiods )
    
B_sequence = np.array(B_sequence)
   



# Global expenditure returns (use data for all the periods)
sc = series[:, 1::]-series[:, 0:-1] # get changes in indicators
scr = sc # isolate instrumentals
success_emp = np.sum(scr>0, axis=1)/scr.shape[1]
maxg = .95
ming = .05
success_emp = (maxg - ming)*(success_emp - success_emp.min())/(success_emp.max() - success_emp.min()) + ming

# Initial factors
params = np.ones(2*N)*.5

print()

max_theo = df['Máx teórico'].values
cc = df['control_corrupcion'].values[0]
rl = df['estado_de_derecho'].values[0]

increment = 100
mean_abs_error = 100
normed_errors = np.ones(2*N)*-1
sample_size = 10
counter = 0
while mean_abs_error > .02:
    
    counter += 1
    alphas_t = params[0:N]
    betas_t = params[N::]
    
    errors = np.array(fobj2(I0, A, R, alphas_t, cc, rl, betas_t, scalar, sample_size, G, success_emp, B_sequence, budget_hash))
    normed_errors = errors/np.array((G-I0).tolist() + success_emp.tolist())
    abs_errors = np.abs(errors)
    abs_normed_errrors = np.abs(normed_errors)
    
    mean_abs_error = np.mean(abs_errors)
    
    params[errors<0] *= np.clip(1-abs_normed_errrors[errors<0], .5, 1)
    params[errors>0] *= np.clip(1+abs_normed_errrors[errors>0], 1, 1.5)
    
    if counter > 20:
        sample_size += increment
        increment += 10
    
    print(ith, mean_abs_error, sample_size, counter,  abs_normed_errrors.max())

print('computing final estimate...')
print()
sample_size = 1000
alphas_est = params[0:N]
betas_est = params[N::]
errors_est = np.array(fobj2(I0, A, R, alphas_est, cc, rl, betas_est, scalar, sample_size, G, success_emp, B_sequence, budget_hash))
errors_alpha = errors_est[0:N]
error_beta = errors_est[N::]

GoF_alpha = 1 - np.abs(errors_alpha)/(G-I0)
GoF_beta = 1 - np.abs(error_beta)/success_emp

betas_final_est = np.zeros(N)
betas_final_est = betas_est
dfc = pd.DataFrame([[alphas_est[i], betas_final_est[i], max_steps, num_years, errors_alpha[i]/scalar, error_beta[i], scalar, min_value, GoF_alpha[i], GoF_beta[i]] \
                    if i==0 else [alphas_est[i], betas_final_est[i], np.nan, np.nan, errors_alpha[i]/scalar, error_beta[i], np.nan, np.nan, GoF_alpha[i], GoF_beta[i]] \
                   for i in range(N)], 
                   columns=['alphas', 'beta', 'steps', 'years', 'error_alpha', 'error_beta', 'scalar', 'min_value', 'GoF_alpha', 'GoF_beta'])
dfc.to_csv(home+'data/parametros.csv', index=False)









