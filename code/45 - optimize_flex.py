'''

Corre el algoritmo de evolución diferenciada para optimizar el gaso solo entre
los indicadores sensibles.

Es necesario correr el script para distintos niveles de aumento del 
presupuesto (x0, x100, y x1000).

'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed

home =  os.getcwd()[:-4]

os.chdir(home+'/code/')
import ppi
from functions import *
 

parallel_processes = 40
sample_size = 100


# Dataset

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
years_forward = 10

# Indicators
N = len(dft)
R = dft.Instrumental.values.copy()
n = R.sum()
I0 = df['2020'].values.copy()*scalar
Imax = np.ones(N)*scalar


# Budget
df_expt = pd.read_csv(home+"data/modeling/budget_target.csv")
annual_average_bud = df_expt[colYears].mean(axis=1).values
Bs_retro_tot = np.tile(annual_average_bud, (years_forward,1)).T
Bs = [[] for g in range(len(df_expt))]
for i, year in enumerate(range(2021, 2020+years_forward+1)):
    for j in range(sub_periods):
        for index, row in df_expt.iterrows():
            Bs[index].append( Bs_retro_tot[index,i]/sub_periods )
Bs0 = np.array(Bs)
usdgs = df_expt.values[:,0]
sdg2index = dict(zip(usdgs, range(len(usdgs))))
sdgs = dft.MetaODS.values
B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])


# Network
A = np.loadtxt(home+"data/modeling/network.csv", delimiter=',')

# Governance
qm = np.ones(n)*dft.Monitoreo.values[0]
rl = np.ones(n)*dft.EstadoDeDerecho.values[0]







dfi_pro = pd.read_csv(home+"data/sims/prospective.csv")
fin_vals_pro = dfi_pro.values[:,years_forward*sub_periods]

dfi_fro = pd.read_csv(home+"data/sims/frontier.csv")
fin_vals_fro = dfi_fro.values[:,years_forward*sub_periods]

convergence = dfi_fro.values[:,1:years_forward*sub_periods+1] >= np.tile(fin_vals_pro, (dfi_fro.values[:,1:years_forward*sub_periods+1].shape[1],1)).T
savings = np.array([12*(years_forward*sub_periods - np.where(conv)[0][0])/sub_periods for conv in convergence])
sensible = np.where( (savings >= 60) & (fin_vals_pro < df.Meta2030*scalar))[0]
targets_sensible = np.where(df_expt.MetaODS.isin(df.MetaODS.values[sensible]).values)[0]
budget_sensible = annual_average_bud[targets_sensible].sum()
indis_sensible = np.where(df.MetaODS.isin(df_expt.MetaODS.values[targets_sensible]))[0]


n_sdgs = len(targets_sensible)

print('Runing model ...')


def fobj2(presu, interest):
    
    # Budget
    fracs = presu/presu.sum()
    Bs = [[] for g in range(len(df_expt))]
    for i, year in enumerate(range(2021, 2020+years_forward+1)):
        for j in range(sub_periods):
            index2=0
            for index, row in df_expt.iterrows():
                if index in targets_sensible:
                    Bs[index].append( (1+interest/100)*fracs[index2]*budget_sensible/sub_periods )
                    index2+=1
                else:
                    Bs[index].append( Bs_retro_tot[index,i]/sub_periods )
    Bs = np.array(Bs)


        
    ## Serial
    sols = [ppi.run_ppi(I0, alphas, betas, A=A, R=R, qm=qm, rl=rl, Imax=Imax, 
                      Bs=Bs, B_dict=B_dict, T=T, scalar=scalar) for sample in range(sample_size)]
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    

    
    aver_Is = np.mean(tsI, axis=0)
    levels =  df.Meta2030.values[indis_sensible]*scalar - aver_Is[:,-1][indis_sensible]
    levels[levels<0] = 0
    error = np.average( levels )
    return error, Bs[:,0]*sub_periods







best_fitness = 1000
popsize = 20
mut=0.8
crossp=0.7

# [0, 100, 1000]
    
interest = 1000
bounds = np.array(list(zip(.0001*np.ones(n_sdgs), .99*np.ones(n_sdgs))))
min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)
dimensions = len(bounds)
pop =  np.random.rand(popsize, dimensions)*.8 + .2
best_sols = []

step = 0
while True:
    print(interest, step)
    
    # fitness = [fobj2(Bs) for Bs in pop] # Serial
    results = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(fobj2)(Bs, interest) for Bs in pop) # Parallel
    fitness, new_budgets = zip(*results)
    best_idx = np.argmin(fitness)
    
    if fitness[best_idx] < best_fitness:
        best_sol = pop[best_idx]
        best_fitness = fitness[best_idx]
        print(best_fitness)
        best_budget = [[b] for b in new_budgets[best_idx]]
        M = np.hstack( ([[c] for c in df_expt.MetaODS], best_budget) )
        df_sol = pd.DataFrame(M, columns=['MetaODS', 'gasto'])
        df_sol.to_csv(home+"data/modeling/optimal_budget_"+str(interest)+".csv", index=False)       

    sorter = np.argsort(fitness)
    survivors = pop[sorter][0:int(len(fitness)/2)].tolist()
    new_pop = survivors.copy()
    
    newPop = []
    for j in range(len(survivors)):
        idxs = [idx for idx in range(len(survivors)) if idx != j]
        a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
        mutant = np.clip(a + mut * (b - c), 10e-12, 1)
        cross_points = np.random.rand(dimensions) < crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimensions)] = True
        trial = np.where(cross_points, mutant, pop[j])
        trial_denorm = min_b + trial * diff
        new_pop.append(trial_denorm)
        
    pop = np.array(new_pop)
    step += 1










































