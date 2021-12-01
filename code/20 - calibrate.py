'''

Calibra los parámetros del modelo usando los datos históricos.

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





df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]





parallel_processes = 10


num_years = len(colYears)
scalar = 100
min_value = 1e-2

sub_periods = 4
T = len(colYears)*sub_periods




# Extract country data
dft = df
df_expt = pd.read_csv(home+"data/modeling/budget_target.csv")

# Indicators
series = dft[colYears].values
N = len(dft)
Imax = np.ones(N)*scalar
R = dft.Instrumental.values.copy()
n = R.sum()
I0, IF = get_conditions(series, scalar, min_value, Imax)
success_rates = get_success_rates(series)


# Budget
Bs = np.tile(df_expt[colYears].mean(axis=1).values, (len(colYears),1)).T
usdgs = df_expt.values[:,0]
sdg2index = dict(zip(usdgs, range(len(usdgs))))
sdgs = dft.MetaODS.values
B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])
Bs = get_dirsbursement_schedule(Bs, B_dict, T)

# Network
A = np.loadtxt(home+"data/modeling/network.csv", delimiter=',')

# Governance
qm = np.ones(n)*dft.Monitoreo.values[0]
rl = np.ones(n)*dft.EstadoDeDerecho.values[0]

# Perform calibration
params = np.ones(2*N)*.5
increment = 100
mean_abs_error = 100
normed_errors = np.ones(2*N)*-1
sample_size = 10
counter = 0
tolerane = .1

GoF_alpha = np.zeros(N)
GoF_beta = np.zeros(N)

while np.sum(GoF_alpha<.9) > 0 or np.sum(GoF_beta<.9) > 0:

    counter += 1
    alphas = params[0:N]
    betas = params[N::]
    
    errors = np.array(compute_error(I0, alphas, betas, A, R, qm, rl, Imax, Bs, B_dict, T, scalar, IF, success_rates, parallel_processes, sample_size))
    normed_errors = errors/np.array((IF-I0).tolist() + success_rates.tolist())
    abs_normed_errrors = np.abs(normed_errors)
    
    mean_abs_error = np.mean(np.abs(errors))
    
    params[errors<0] *= np.clip(1-abs_normed_errrors[errors<0], .25, 1)
    params[errors>0] *= np.clip(1+abs_normed_errrors[errors>0], 1, 1.5)
    
    errors_alpha = errors[0:N]
    errors_beta = errors[N::]
    GoF_alpha = 1 - np.abs(errors_alpha)/(IF-I0)
    GoF_beta = 1 - np.abs(errors_beta)/success_rates
    
    if counter > 20:
        sample_size += increment
        increment += 10
    
    print(mean_abs_error, sample_size, counter,  abs_normed_errrors.max(), np.min(GoF_alpha.tolist()+GoF_beta.tolist()))

print('computing final estimate...')
print()
sample_size = 1000
alphas_est = params[0:N]
betas_est = params[N::]
errors_est = np.array(compute_error(I0, alphas_est, betas_est, A, R, qm, rl, Imax, Bs, B_dict, T, scalar, IF, success_rates, parallel_processes, sample_size))
errors_alpha = errors_est[0:N]
errors_beta = errors_est[N::]

GoF_alpha = 1 - np.abs(errors_alpha)/(IF-I0)
GoF_beta = 1 - np.abs(errors_beta)/success_rates

betas_final_est = np.zeros(N)
betas_final_est = betas_est
dfc = pd.DataFrame([[alphas_est[i], betas_final_est[i], T, num_years, errors_alpha[i]/scalar, errors_beta[i], scalar, min_value, GoF_alpha[i], GoF_beta[i]] \
                    if i==0 else [alphas_est[i], betas_final_est[i], np.nan, np.nan, errors_alpha[i]/scalar, errors_beta[i], np.nan, np.nan, GoF_alpha[i], GoF_beta[i]] \
                    for i in range(N)], 
                    columns=['alpha', 'beta', 'T', 'years', 'error_alpha', 'error_beta', 'scalar', 'min_value', 'GoF_alpha', 'GoF_beta'])
dfc.to_csv(home+'data/modeling/parameters.csv', index=False)


 























































