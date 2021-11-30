'''
Este script corre el algoritmo de evolucion diferenciada para encontrar una
redistribucion presupuestal optima bajo el escenario fiscal conservador y con
una tasa de crecimiento a anual del presupuesto de 20% bajo el escenario Covid
del reporte. 

El algoritmo corre iterativamente de forma perpetua y guarda la distribucion 
optima en cada iteracion. Se recomienda detener el algoritmo despues de 200 
iteraciones pues es poco probable encontrar mejores soluciones despues de este umbral.

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
    outputs = ppi.run_ppi(I0, A=A, R=R, alpha=alpha, cc=cc, rl=rl,
            betas=betas, get_gammas=True, scalar=scalar, B_sequence=B_sequence, budget_hash=budget_hash,
            max_theo=max_theo)
    tsI, tsC, tsF, tsP, tsD, tsS, ticks, H, gammas = outputs
    return tsI[:,-1]





# Dataset
df = pd.read_csv(home+"data/base_final.csv")
colYears = [col for col in df.columns if col.isnumeric()]

parallel_processes = 15
sample_size = 100


A = np.loadtxt(home+'data/red/A.csv', dtype=float, delimiter=",")  
series = df[colYears].values
N = len(df)

R = df['Instrumental'].values==1


# Initial factors
adf = pd.read_csv(home+"data/parametros.csv")
alphas = adf.alphas.values.copy()
max_steps = int(adf.steps.values[0])
betas = adf.beta.values.copy()
num_years = adf.years.values[0]
scalar = adf.scalar.values[0]
min_value = adf.min_value.values[0]



indis = df.Abreviatura.values
budget_hash = dict([ (i, ods[(ods!='nan') & (ods != '10.0')].astype(float).astype(int).tolist()) for i, ods in enumerate(df[['ODS1', 'ODS2']].values.astype(str)) ])


dfb0 = pd.read_csv(home+"data/base_presupuesto.csv")
dfb = dfb0.groupby('ODS').sum()
programs = sorted(dfb.index.values.astype(float).astype(str))

I0 = df['2020'].values.copy()*scalar
reducs = pd.read_excel(home+'data/raw/covid.xlsx')
I0[df.Código.isin(reducs.Código)] *= (1 - reducs.Reducción.values/100)

interest = 0.2

max_theo = df['Máx teórico'].values*scalar
cc = df['control_corrupcion'].values[0]
rl = df['estado_de_derecho'].values[0]

output_data = []


bud_fracs = []
for index, row in dfb0.iterrows():
    if index > 0:
        if row.ODS == dfb0.loc[index-1].ODS:
            bud_fracs.append( np.min([1, np.mean(dfb0.loc[index-1][colYears].values/row[colYears].values)]) )
        elif row.ODS != dfb0.loc[index+1].ODS:
            bud_fracs.append(0)
halves = np.ones(len(bud_fracs))*.5
min_fracs = np.max([halves, 1-np.array(bud_fracs)], axis=0)
min_fracs[0] = 1

B_sequence = np.zeros((len(dfb), 10*int(max_steps/num_years)))


dfw = pd.read_excel(home+'data/raw/ods_pesos_covid.xlsx')
wdict = dict(dfw.values)
weights = np.array([wdict[ods] for ods in df.ODS1])



n_sdgs, n_periods = B_sequence.shape

print('Runing model ...')

def fobj2(presu):
    B_seq = B_sequence.copy()
    bench = dfb['2020'].values.copy()/(max_steps/num_years)
    for i in range(B_sequence.shape[1]):
        prop = presu/presu.sum() 
        B_seq[:,i] = min_fracs*bench + (bench.sum()-(min_fracs*bench).sum())*prop
        if i%int(max_steps/num_years)==0:
            bench *= (1+interest)

    fin_indis = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)(I0, A, R, alphas, cc, rl, betas, scalar, B_seq, budget_hash, max_theo) for itera in range(sample_size))
    levels = df.Meta.values*100 - np.mean(fin_indis, axis=0)
    levels[levels<0] = 0
    error = np.mean( np.abs(levels)*weights )
    return error



    
dfb = pd.read_csv(home+"data/base_presupuesto.csv")
dfb = dfb.groupby('ODS').sum()
bench = dfb['2020'].values/(max_steps/num_years)
B_sequence_ref = np.zeros((len(dfb), int(10*max_steps/num_years)))
for i in range(B_sequence_ref.shape[1]):
    B_sequence_ref[:,i] = bench
    if i%int(max_steps/num_years)==0:
        bench *= (1+interest)
    
    
    
fin_indis = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)(I0, A, R, alphas, cc, rl, betas, scalar, B_sequence_ref, budget_hash, max_theo) for itera in range(1000))
levels = df.Meta.values*100 - np.mean(fin_indis, axis=0)
levels[levels<0] = 0
best_fitness = np.mean( np.abs(levels)*weights )
print(best_fitness)
    

popsize = 24
njobs = 2
mut=0.8
crossp=0.7

bounds = np.array(list(zip(.0001*np.ones(n_sdgs), .99*np.ones(n_sdgs))))
min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)
dimensions = len(bounds)
pop =  np.random.rand(popsize, dimensions)*.8 + .2
pop[0] = 1-min_fracs
pop[1] = B_sequence_ref[:,0]/B_sequence_ref[:,0].sum()
best_sols = []

step = 0
while True:
    print(step)
    
    fitness = [fobj2(Bs) for Bs in pop]
    best_idx = np.argmin(fitness)
    
    if fitness[best_idx] < best_fitness:
        best_sol = pop[best_idx]
        best_fitness = fitness[best_idx]
        print(best_fitness)
        best_sols.append(best_sol.tolist()+[best_fitness])
        df_sol = pd.DataFrame(best_sols, columns=list(range(len(best_sol)))+['fitness'])
        df_sol.to_csv(home+"data/optimal_budget_covid2.csv", index=False)       

    sorter = np.argsort(fitness)
    survivors = pop[sorter][0:int(len(fitness)/2)].tolist()
    new_pop = survivors.copy()
    
    newPop = []
    for j in range(len(survivors)):
        idxs = [idx for idx in range(len(survivors)) if idx != j]
        a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
        mutant = np.clip(a + mut * (b - c), 0, 1)
        cross_points = np.random.rand(dimensions) < crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimensions)] = True
        trial = np.where(cross_points, mutant, pop[j])
        trial_denorm = min_b + trial * diff
        new_pop.append(trial_denorm)
        
    pop = np.array(new_pop)
    step += 1










































