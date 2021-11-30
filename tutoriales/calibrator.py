import numpy as np
import pandas as pd
from joblib import Parallel, delayed


import requests
url = 'https://raw.githubusercontent.com/oguerrer/IPP_Lima/main/code/ppi.py'
r = requests.get(url)
with open('ppi.py', 'w') as f:
    f.write(r.text)
import ppi





def run_ppi_parallel(I0, A, R, alpha, cc, rl, betas, scalar, B_sequence, budget_hash):
    outputs = ppi.run_ppi(I0, A=A, R=R, alpha=alpha, cc=cc, rl=rl,
            betas=betas, get_gammas=True, scalar=scalar, B_sequence=B_sequence, budget_hash=budget_hash)
    tsI, tsC, tsF, tsP, tsD, tsS, times, H, gammas = outputs
    return (tsI[:,-1], gammas)



def fobj2(I0, A, R, alpha, cc, rl, betas, scalar, sample_size, IF, success_emp, B_sequence, budget_hash, parallel_processes):
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
            (I0, A, R, alpha, cc, rl, betas, scalar, B_sequence, budget_hash) for itera in range(sample_size)))
    FIs = []
    gammas = []
    for sol in sols:
        FIs.append( sol[0] )
        for gamma in sol[1]:
            gammas.append( gamma )

    mean_indis = np.mean(FIs, axis=0)
    error_alpha = IF - mean_indis
    mean_gamma = np.mean(gammas, axis=0)
    error_beta = success_emp - mean_gamma

    return error_alpha.tolist() + error_beta.tolist()







def calibrate(I0, A, R, cc, rl, scalar, IF, success_emp, B_sequence, budget_hash, num_years, max_steps, min_value, tolerance=.05, parallel_processes=2):

    N = len(I0)
    params = np.ones(2*N)*.5

    increment = 100
    mean_abs_error = 100
    normed_errors = np.ones(2*N)*-1
    sample_size = 10
    counter = 0
    while mean_abs_error > tolerance:
        
        counter += 1
        alphas_t = params[0:N]
        betas_t = params[N::]
        
        errors = np.array(fobj2(I0, A, R, alphas_t, cc, rl, betas_t, scalar, sample_size, IF, success_emp, B_sequence, budget_hash, parallel_processes))
        normed_errors = errors/np.array((IF-I0).tolist() + success_emp.tolist())
        abs_errors = np.abs(errors)
        abs_normed_errrors = np.abs(normed_errors)
        
        mean_abs_error = np.mean(abs_errors)
        
        params[errors<0] *= np.clip(1-abs_normed_errrors[errors<0], .5, 1)
        params[errors>0] *= np.clip(1+abs_normed_errrors[errors>0], 1, 1.5)
        
        if counter > 20:
            sample_size += increment
            increment += 10
        
        print('iteración:', counter, '; muestras:', sample_size, '; error:', mean_abs_error)
    
    print('calculando la bondad de ajuste...')
    print()
    sample_size = 1000
    alphas_est = params[0:N]
    betas_est = params[N::]
    errors_est = np.array(fobj2(I0, A, R, alphas_est, cc, rl, betas_est, scalar, sample_size, IF, success_emp, B_sequence, budget_hash, parallel_processes))
    errors_alpha = errors_est[0:N]
    error_beta = errors_est[N::]
    
    GoF_alpha = 1 - np.abs(errors_alpha)/(IF-I0)
    GoF_beta = 1 - np.abs(error_beta)/success_emp
    
    dfc = pd.DataFrame([[alphas_est[i], betas_est[i], max_steps, num_years, errors_alpha[i]/scalar, error_beta[i], scalar, min_value, GoF_alpha[i], GoF_beta[i]] \
                        if i==0 else [alphas_est[i], betas_est[i], np.nan, np.nan, errors_alpha[i]/scalar, error_beta[i], np.nan, np.nan, GoF_alpha[i], GoF_beta[i]] \
                        for i in range(N)], 
                        columns=['alphas', 'beta', 'steps', 'years', 'error_alpha', 'error_beta', 'scalar', 'min_value', 'GoF_alpha', 'GoF_beta'])
    return dfc
    





