import numpy as np
import pandas as pd
from joblib import Parallel, delayed


import requests
url = 'https://raw.githubusercontent.com/oguerrer/IPP_Colombia/main/code/ppi.py'
r = requests.get(url)
with open('ppi.py', 'w') as f:
    f.write(r.text)
import ppi





def get_conditions(series, scalar, min_value, Imax=None):
    I0 = series[:,0]*scalar
    IF = series[:,-1]*scalar
    gaps = IF - I0
    IF[gaps < min_value] = I0[gaps < min_value] + min_value
    if Imax is not None:
        invalid = IF >= Imax
        IF[invalid] = Imax[invalid]
        I0[invalid] = IF[invalid]-min_value
    return I0, IF



def get_success_rates(series):
    sc = series[:, 1::]-series[:, 0:-1] # get changes in indicators
    success_rates = np.sum(sc>0, axis=1)/sc.shape[1] # compute rates of success
    success_rates = .9*(success_rates-success_rates.min())/(success_rates.max()-success_rates.min()) + .05
    return success_rates



def get_dirsbursement_schedule(Bs, B_dict, T):
    programs = sorted(list(set([item for subl in list(B_dict.values()) for item in subl])))
    B_sequence = [[] for program in programs]
    subperiods = int(T/Bs.shape[1])
    for i, program in enumerate(programs):
        for period in range(Bs.shape[1]):
            for subperiod in range(subperiods):
                B_sequence[i].append( Bs[i,period]/subperiods )
    B_sequence = np.array(B_sequence)
    return B_sequence




def run_ppi_parallel(I0, alphas, betas, A, R, qm, rl, Imax, Bs, B_dict, T, scalar, frontier=None):
    outputs = ppi.run_ppi(I0, alphas, betas, A=A, R=R, qm=qm, rl=rl, Imax=Imax, 
                      Bs=Bs, B_dict=B_dict, T=T, scalar=scalar, frontier=frontier)
    return outputs




def compute_error(I0, alphas, betas, A, R, qm, rl, Imax, Bs, B_dict, T, scalar, IF, success_rates, parallel_processes, sample_size):
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
            (I0, alphas, betas, A, R, qm, rl, Imax, Bs, B_dict, T, scalar) for itera in range(sample_size)))
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    I_hat = np.mean(tsI, axis=0)[:,-1]
    gamma_hat = np.mean(tsG, axis=0).mean(axis=1)
    error_alpha = IF - I_hat
    error_beta = success_rates - gamma_hat
    return error_alpha.tolist() + error_beta.tolist()








def calibrate(I0, A, R, qm, rl,  Bs, B_dict, T, scalar, IF, Imax, success_rates, num_years, min_value, tolerance=.9, parallel_processes=2):

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
        
        errors = np.array(compute_error(I0=I0, alphas=alphas, betas=betas, A=A, R=R, qm=qm, rl=rl,  
                                Bs=Bs, B_dict=B_dict, T=T, scalar=scalar, IF=IF, Imax=Imax, success_rates=success_rates, 
                                sample_size=sample_size, parallel_processes=parallel_processes))
        normed_errors = errors/np.array((IF-I0).tolist() + success_rates.tolist())
        abs_normed_errrors = np.abs(normed_errors)
        
        mean_abs_error = np.mean(np.abs(errors))
        
        params[errors<0] *= np.clip(1-abs_normed_errrors[errors<0], .25, 1)
        params[errors>0] *= np.clip(1+abs_normed_errrors[errors>0], 1, 1.5)
        
        errors_alpha = errors[0:N]
        errors_beta = errors[N::]
        GoF_alpha = 1 - np.abs(errors_alpha)/(IF-I0)
        GoF_beta = 1 - np.abs(errors_beta)/success_rates
        
        if counter > 50:
            sample_size += increment
        
        print( counter, np.min(GoF_alpha.tolist()), np.min(GoF_beta.tolist()) )
    
    print('computing final estimate...')
    print()
    sample_size = 1000
    alphas_est = params[0:N]
    betas_est = params[N::]
    errors_est = np.array(compute_error(I0=I0, alphas=alphas_est, betas=betas_est, A=A, R=R, qm=qm, rl=rl,  
                                Bs=Bs, B_dict=B_dict, T=T, scalar=scalar, IF=IF, Imax=Imax, success_rates=success_rates, 
                                sample_size=sample_size, parallel_processes=parallel_processes))
    errors_alpha = errors_est[0:N]
    errors_beta = errors_est[N::]
    
    GoF_alpha = 1 - np.abs(errors_alpha)/(IF-I0)
    GoF_beta = 1 - np.abs(errors_beta)/success_rates
    
    dfc = pd.DataFrame([[alphas_est[i], betas_est[i], T, num_years, errors_alpha[i]/scalar, errors_beta[i], scalar, min_value, GoF_alpha[i], GoF_beta[i]] \
                    if i==0 else [alphas_est[i], betas_est[i], np.nan, np.nan, errors_alpha[i]/scalar, errors_beta[i], np.nan, np.nan, GoF_alpha[i], GoF_beta[i]] \
                    for i in range(N)], 
                    columns=['alpha', 'beta', 'T', 'years', 'error_alpha', 'error_beta', 'scalar', 'min_value', 'GoF_alpha', 'GoF_beta'])
    return dfc
    






























