import numpy as np
import pandas as pd
from joblib import Parallel, delayed


import requests
url = 'https://raw.githubusercontent.com/oguerrer/IPP_Lima/main/code/ppi.py'
r = requests.get(url)
with open('ppi.py', 'w') as f:
    f.write(r.text)
import ppi




def run_ppi_parallel(I0, alphas, betas, A, R, qm, rl, Bs, B_dict, T, scalar, frontier=None):
    outputs = ppi.run_ppi(I0=I0, alphas=alphas, betas=betas, A=A, R=R, qm=qm, rl=rl, 
                      Bs=Bs, B_dict=B_dict, T=T, scalar=scalar, frontier=frontier)
    tsI, tsC, tsF, tsP, tsD, tsS, times, H, gammas = outputs
    return (tsI[:,-1], gammas)



def fobj2(I0, alphas, betas, A, R, qm, rl,  Bs, B_dict, T, scalar, IF, success_emp, sample_size, parallel_processes):
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
            (I0=I0, alphas=alphas, betas=betas, A=A, R=R, qm=qm, rl=rl, 
             Bs=Bs, B_dict=B_dict, T=T, scalar=scalar) for itera in range(sample_size)))
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







def calibrate(I0, alphas, betas, A, R, qm, rl,  Bs, B_dict, T, scalar, IF, success_emp, num_years, max_steps, min_value, tolerance=.9, parallel_processes=2):

    # Perform calibration
    N = len(I0)
    params = np.ones(2*N)*.5
    increment = 100
    mean_abs_error = 100
    normed_errors = np.ones(2*N)*-1
    sample_size = 10
    counter = 0
    
    GoF_alpha = np.zeros(N)
    GoF_beta = np.zeros(N)
    
    while np.sum(GoF_alpha<tolerance) > 0 or np.sum(GoF_beta<tolerance) > 0:
    
        counter += 1
        alphas = params[0:N]
        betas = params[N::]
        
        errors = np.array(fobj2(I0=I0, alphas=alphas, betas=betas, A=A, R=R, qm=qm, rl=rl,  
                                Bs=Bs, B_dict=B_dict, T=T, scalar=scalar, IF=IF, success_emp=success_emp, 
                                sample_size=sample_size, parallel_processes=parallel_processes))
        normed_errors = errors/np.array((IF-I0).tolist() + success_emp.tolist())
        abs_normed_errrors = np.abs(normed_errors)
        
        mean_abs_error = np.mean(np.abs(errors))
        
        params[errors<0] *= np.clip(1-abs_normed_errrors[errors<0], .25, 1)
        params[errors>0] *= np.clip(1+abs_normed_errrors[errors>0], 1, 1.5)
        
        errors_alpha = errors[0:N]
        errors_beta = errors[N::]
        GoF_alpha = 1 - np.abs(errors_alpha)/(IF-I0)
        GoF_beta = 1 - np.abs(errors_beta)/success_emp
        
        if counter > 20:
            sample_size += increment
            increment += 10
        
        print(mean_abs_error, sample_size, counter,  abs_normed_errrors.max(), np.min(GoF_alpha.tolist()+GoF_beta.tolist()))
    
    print('computing final estimate...')
    print()
    sample_size = 1000
    alphas_est = params[0:N]
    betas_est = params[N::]
    errors_est = np.array(fobj2(I0=I0, alphas=alphas_est, betas=betas_est, A=A, R=R, qm=qm, rl=rl,  
                                Bs=Bs, B_dict=B_dict, T=T, scalar=scalar, IF=IF, success_emp=success_emp, 
                                sample_size=sample_size, parallel_processes=parallel_processes))
    errors_alpha = errors_est[0:N]
    errors_beta = errors_est[N::]
    
    GoF_alpha = 1 - np.abs(errors_alpha)/(IF-I0)
    GoF_beta = 1 - np.abs(errors_beta)/success_emp
    
    dfc = pd.DataFrame([[alphas_est[i], betas_est[i], max_steps, num_years, errors_alpha[i]/scalar, errors_beta[i], scalar, min_value, GoF_alpha[i], GoF_beta[i]] \
                        if i==0 else [alphas_est[i], betas_est[i], np.nan, np.nan, errors_alpha[i]/scalar, errors_beta[i], np.nan, np.nan, GoF_alpha[i], GoF_beta[i]] \
                        for i in range(N)], 
                        columns=['alphas', 'beta', 'steps', 'years', 'error_alpha', 'error_beta', 'scalar', 'min_value', 'GoF_alpha', 'GoF_beta'])
    return dfc
    






























