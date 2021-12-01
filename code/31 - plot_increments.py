'''

Grafica los resultados de las simulaciones contrafactuales de aumentos presupuestales.

'''


import matplotlib.pyplot as plt
import numpy as np
import os, copy, re, csv
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as st


home =  os.getcwd()[:-4]

path = home+'/figuras/'


df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = np.array([c for c in df.columns if c.isnumeric()])

file = open(home+"/data/misc/sdg_colors.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()






# Parameters
df_params = pd.read_csv(home+"data/modeling/parameters.csv")
alphas = df_params.alpha.values.copy()
betas = df_params.beta.values.copy()
T = int(df_params['T'].values[0])
scalar = df_params.scalar.values[0]
min_value = df_params.min_value.values[0]
num_years = df_params.years.values[0]
sub_periods = int(T/num_years)






dfi = pd.read_csv(home+"data/sims/prospective.csv")
all_vals = dfi.values[:,1::]

data_out = [df.Abreviatura.tolist()]

gaps = 100*(df.Meta2030.values*scalar - all_vals[:,0])/(df.Meta2030.values*scalar)
gapsc = np.max([np.zeros(len(df)) , 100*(df.Meta2030.values*100 - all_vals[:,sub_periods*10+1])/(df.Meta2030.values*100)], axis=0)
all_gaps_t = []
for interest in list(range(5, 21, 5)):
    dft = pd.read_csv(home+"data/sims/increment_"+str(interest)+".csv")
    all_vals_t = dft.values[:,1::]
    gaps_t = np.max([np.zeros(len(df)) , 100*(df.Meta2030.values*scalar - all_vals_t[:,sub_periods*10+1])/(df.Meta2030.values*scalar)], axis=0)
    all_gaps_t.append(gaps_t)
all_gaps_t = np.array(all_gaps_t)
plt.figure(figsize=(12,3.5))
plt.plot(-1000, -1000, 'o', mfc='none', mec='black', markersize=5, label='presupuesto de 2020 proyectado')
plt.plot(-1000, -1000, '.k', markersize=2, label='tasa de crecimiento aumentada')
labels = []
for index, row in df.iterrows():
    ODS1 = row.ODS
    if row.Instrumental==0:
        plt.plot( index, 0, '.', markersize=16, mec='w', mfc=colors_sdg[ODS1])
    else:
        plt.plot( index, 0, 'p', markersize=10, mec='w', mfc=colors_sdg[ODS1])
    labels.append(row.Abreviatura)
data_out.append((100*(gaps-gapsc)/gaps).tolist())
for gaps_t in all_gaps_t:
    data_out.append((100*(gaps-gaps_t)/gaps).tolist())
    plt.plot( range(len(gaps)), 100*(gaps-gaps_t)/gaps, '.k', markersize=2,)
plt.plot( range(len(gaps)), 100*(gaps-gapsc)/gaps, 'o', mec='k', mfc='none', markersize=5,)
plt.xlim(-1, len(gaps))
plt.ylim(-5, 140)
plt.gca().set_xticks(range(len(gaps)))
plt.gca().set_xticklabels(labels, rotation=90, fontsize=7)
plt.gca().set_yticks(range(0, 101, 25))
plt.ylabel('cierre de brecha (%)', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.legend(fontsize=12, ncol=3)
plt.tight_layout()
plt.savefig(path+'brechas_reduc_2030.pdf')
plt.show()

data_out = np.array(data_out).T








plt.figure(figsize=(6,4))
params = []
indices = []
for index, row in df.iterrows():
    serie = all_gaps_t.T[index]
    closure_ini = 100*(gaps[index]-gapsc[index])/gaps[index]
    closures = np.array([closure_ini] + (100*(gaps[index]-serie)/gaps[index]).tolist())
    plt.plot([0]+list(range(5, 21, 5)), closures-closure_ini, color=colors_sdg[row.ODS], linewidth=2)
    indices.append(index)
    params.append( (closures[-1]-closure_ini)/20 )
# plt.xlim(.5, len(x)-.5)
# plt.ylim(-5, 125)
plt.xticks([0]+list(range(5, 21, 5)))
plt.xlabel('aumento de tasa crecimiento presupuestal (%)', fontsize=14)
plt.ylabel('aumento en reducción de brecha', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(path+'aumento_curves.pdf')
plt.show()












plt.figure(figsize=(6,4))
for index, row in df.iterrows():
    param = params[index]
    level = row['2020']*100
    if row.Instrumental==0:
        plt.plot(level, 1*param, '.', mfc=colors_sdg[row.ODS], mec='w', markersize=20)
    else:
        plt.plot(level, 1*param, 'p', mfc=colors_sdg[row.ODS], mec='w', markersize=12)
# plt.xlim(1, len(x))
# plt.ylim(-1, 10)
plt.xlabel('nivel del indicador en 2020', fontsize=14)
plt.ylabel('elasticidad arco', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(path+'aumento_params.pdf')
plt.show()











data_out = []

sarg = np.argsort(params)
plt.figure(figsize=(12,3.5))
i=0
labels = []
for arg in sarg:
    row = df.loc[indices[arg]]
    param = params[arg]
    if param > 0:
        labels.append(row.Abreviatura)
        data_out.append([row.Abreviatura, param])
        if i%2==0:
            plt.plot([i,i], [-10, param], '-', color='grey', linewidth=1.)
        else:
            plt.plot([i,i], [-10, param], '--', color='black', linewidth=1.)
        if row.Instrumental==0:
            plt.plot(i, param, '.', mfc=colors_sdg[row.ODS], mec='w', markersize=30)
        else:
            plt.plot(i, param, 'p', mfc=colors_sdg[row.ODS], mec='w', markersize=15)
        i+=1
plt.ylim(-.1, 1)
plt.xlim(-1, len(labels))
plt.gca().set_xticks(range(len(labels)))
plt.gca().set_xticklabels(labels, rotation=90, fontsize=10)
plt.ylabel('elasticidad arco', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(path+'aumento_indis.pdf')
plt.show()







dft = pd.read_csv(home+"data/sims/increment_20.csv")
all_vals_t = dft.values[:,1::]

on_time=[]
late=[]
unfeasible=[]
feasible=[]
i=0
for index, row in df.iterrows():
    meta = 100*row.Meta2030
    reaches = np.where(all_vals_t[i] >= meta)[0]
    if len(reaches) > 0 and reaches[0]/sub_periods <= 10:
        on_time.append(index)
        feasible.append(index)
    elif len(reaches) > 0 and reaches[0]/sub_periods <= 20:
        late.append(index)
        feasible.append(index)
    else:
        unfeasible.append(index)
    i+=1

fig = plt.figure(figsize=(4.5,4.5))
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
plt.savefig(path+'dona_convergencia_doble.pdf')
plt.show()

























































