'''

Corre la reasignación presupuestal del x% de los indicadores insensibles a los
sensibles bajo distintos niveles de resignación (10 y 20%).

Grafica los resultados.

'''


import matplotlib.pyplot as plt
import numpy as np
import os, csv
import pandas as pd
from joblib import Parallel, delayed

home =  os.getcwd()[:-4]

os.chdir(home+'/code/')
import ppi
from functions import *


path = home+'/figuras/'





parallel_processes = 40
sample_size = 1000


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
years_forward = 20
T = years_forward*sub_periods

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


file = open(home+"/data/misc/sdg_colors.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()




for trans in [10, 20]:

# trans = 10

    
    # Budget
    df_expt = pd.read_csv(home+"data/modeling/budget_target.csv")
    annual_average_bud = df_expt[colYears].mean(axis=1).values
    Bs_retro_tot = np.tile(annual_average_bud, (years_forward,1)).T
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
    
    
    
    dfi_pro = pd.read_csv(home+"data/sims/prospective.csv")
    fin_vals_pro = dfi_pro.values[:,years_forward*sub_periods]
    
    dfi_fro = pd.read_csv(home+"data/sims/frontier.csv")
    fin_vals_fro = dfi_fro.values[:,years_forward*sub_periods]
    
    convergence = dfi_fro.values[:,1:years_forward*sub_periods+1] >= np.tile(fin_vals_pro, (dfi_fro.values[:,1:years_forward*sub_periods+1].shape[1],1)).T
    savings = np.array([12*(years_forward*sub_periods - np.where(conv)[0][0])/sub_periods for conv in convergence])
    sensible = np.where( (savings >= 60) & (fin_vals_pro < df.Meta2030*scalar))[0]
    insensible = np.where( ~((savings >= 60) & (fin_vals_pro < df.Meta2030*scalar)))[0]
    targets_sensible = np.where(df_expt.MetaODS.isin(df.MetaODS.values[sensible]).values)[0]
    targets_insensible = np.where(df_expt.MetaODS.isin(df.MetaODS.values[insensible]).values)[0]
    
    
    
    
    transfer = np.sum(Bs[targets_insensible] * trans/100)
    Bs[targets_insensible] = Bs[targets_insensible]*(100 - trans/100)
    Bs[targets_sensible] += transfer/Bs[targets_sensible].size
    
    
    
    
    # Simulation
    
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
                (I0, alphas, betas, A, R, qm, rl, Imax, Bs, B_dict, T, scalar) for itera in range(sample_size)))
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    aver_Is = np.mean(tsI, axis=0)
    aver_Ps = np.mean(tsP, axis=0)
    
    
    
    dfi = pd.DataFrame(np.hstack([[[c] for c in df.Abreviatura], aver_Is]), columns=['Abreviatura']+list(range(aver_Is.shape[1])))
    dfi.to_csv(home+"data/sims/transfer_"+str(trans)+".csv", index=False)
    
    
    
    
    
    
    all_vals = dfi.values[:,1::].astype(float)
    
    
    on_time = []
    late = []
    unfeasible = []
    feasible = []
    i=0
    for index, row in df.iterrows():
        ODS1 = row.ODS
        meta = 100*row.Meta2030
        reaches = np.where(all_vals[i] >= meta)[0]
    
        if len(reaches) > 0 and reaches[0]/sub_periods <= 10:
            on_time.append(index)
            feasible.append(index)
        elif len(reaches) > 0 and reaches[0]/sub_periods <= 20:
            late.append(index)
            feasible.append(index)
        else:
            unfeasible.append(index)
        i+=1
    
    
    
    
    
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    width = 0.3
    
    cm = plt.get_cmap("tab20c")
    cout = cm(np.arange(3)*4)
    pie, texts, pcts = ax.pie([len(on_time), len(late), len(unfeasible)], radius=1-width, startangle=90, counterclock=False,
                              colors=['lightgrey', 'grey', 'black'], autopct='%.0f%%', pctdistance=0.79)
    plt.setp( pie, width=width, edgecolor='white')
    plt.setp(pcts[0], color='black')
    plt.setp(pcts[1], color='black')
    plt.setp(pcts[2], color='white')
    ax.legend(pie, ['menos de\n10 años', '10 a 20 años', 'más de\n20 años'],
              loc="center",
              bbox_to_anchor=(.25, .5, 0.5, .0),
              fontsize=8,
              frameon=False
              )
    
    cin = [colors_sdg[df.loc[c].ODS] for c in on_time] + [colors_sdg[df.loc[c].ODS] for c in late] + [colors_sdg[df.loc[c].ODS] for c in unfeasible]
    labels = [df.loc[c].Abreviatura for c in on_time] + [df.loc[c].Abreviatura for c in late] + [df.loc[c].Abreviatura for c in unfeasible]
    pie2, _ = ax.pie(np.ones(len(df)), radius=1, colors=cin, labels=labels, rotatelabels=True, shadow=False, counterclock=False,
                     startangle=90, textprops=dict(va="center", ha='center', rotation_mode='anchor', fontsize=5), 
                     labeldistance=1.17)
    plt.setp( pie2, width=width, edgecolor='none')
    plt.tight_layout()
    plt.savefig(path+'dona_convergencia_transfer_'+str(trans)+'.pdf')
    plt.show()
    
























