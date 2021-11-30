'''
Este script genera la figura 12.

'''

import matplotlib.pyplot as plt
import numpy as np
import os, copy, re, csv
import pandas as pd


home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/base_final.csv")
colYears = np.array([c for c in df.columns if c.isnumeric()])

file = open(home+"/data/color_codes.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()



# Initial factors
adf = pd.read_csv(home+"data/parametros.csv")
alphas = adf.alphas.values.copy()
max_steps = int(adf.steps.values[0])
betas = adf.beta.values.copy()
num_years = adf.years.values[0]
scalar = adf.scalar.values[0]
min_value = adf.min_value.values[0]

R = df.Instrumental==1

dfi = pd.read_csv(home+"data/sims/prospective.csv")
all_vals = dfi.values


dfc = pd.read_csv(home+"data/sims/covid.csv")
all_vals_c = dfc.values


dfw = pd.read_excel(home+'data/raw/ods_pesos_covid.xlsx')
wdict = dict(dfw.values)
weights = np.array([wdict[ods] for ods in df.ODS1])


dfbo = pd.read_csv(home+"data/base_presupuesto.csv")
dfb = dfbo.groupby('ODS').sum()


bud_fracs = []
for index, row in dfbo.iterrows():
    if index > 0:
        if row.ODS == dfbo.loc[index-1].ODS:
            bud_fracs.append( np.min([1, np.mean(dfbo.loc[index-1][colYears].values/row[colYears].values)]) )
        elif row.ODS != dfbo.loc[index+1].ODS:
            bud_fracs.append(0)
halves = np.ones(len(bud_fracs))*.5
min_fracs0 = np.max([halves, 1-np.array(bud_fracs)], axis=0)
min_fracs0[0] = 1



dfb0 = pd.read_csv(home+"data/optimal_budget_covid2.csv")
bdf0 = dfb0.values[-1,0:-1]/dfb0.values[-1,0:-1].sum()


bench = dfb['2020'].values/max_steps
B_seq0 = min_fracs0*bench + (bench.sum()-(min_fracs0*bench).sum())*bdf0




interest = 0.095
bench = dfb['2020'].values/(max_steps/num_years)
B_sequence_ref = np.zeros((len(dfb), int(10*max_steps/num_years)))
for i in range(B_sequence_ref.shape[1]):
    B_sequence_ref[:,i] = bench
    if i%int(max_steps/num_years)==0:
        bench *= (1+interest)
        
        
        


interest = .2
dfbo = pd.read_csv(home+"data/optimal_budget_covid2.csv")
bdf = dfbo.values[-1,0:-1]/dfbo.values[-1,0:-1].sum()
dfb0 = pd.read_csv(home+"data/base_presupuesto.csv")
dfb = dfb0.groupby('ODS').sum()
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
bench = dfb['2020'].values/(max_steps/num_years)
B_sequence = np.zeros((len(dfb), int(10*max_steps/num_years)))
for i in range(B_sequence.shape[1]):
    B_sequence[:,i] = min_fracs*bench + (bench.sum()-(min_fracs*bench).sum())*bdf
    if i%int(max_steps/num_years)==0:
        bench *= (1+interest)






changes = 100*(B_sequence.sum(axis=1)-B_sequence_ref.sum(axis=1))/B_sequence_ref.sum(axis=1)
n_sdgs = len(np.unique(dfb.index.values))
sargll = np.argsort(np.abs(changes))
rankll = np.argsort(sargll)

sarg = np.argsort(np.abs(changes))
rank = np.argsort(sarg)

plt.figure(figsize=(6,4))
for i, ods in enumerate(sorted(np.unique(dfb.index.values))):
    plt.arrow( i, 0, 0, np.sign(changes[i])*(rankll[i]+1), width=.5, 
              head_width=.7, head_length=1.0, color=colors_sdg[ods] )
    text = '{:.2f}'.format(changes[i])+'%'
    align = 1.5
    if changes[i] < 0:
        align *= -1
    plt.text( i, np.sign(changes[i])*(rankll[i]+1)+align, text, ha='center', va='center')
plt.ylim(0, max(rankll)+5)
plt.gca().set_yticks([])
plt.gca().set_xticks(range(len(dfb)))
plt.gca().set_xticklabels(sorted(np.unique(dfb.index.values)))
plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('ODS', fontsize=14)
plt.tight_layout()
plt.savefig(home+'/figuras/flechas_covid2.pdf')
plt.show()

















dfi = pd.read_csv(home+"data/sims/prospective.csv")
M = np.tile(df.Meta.values*100, (dfi.values.shape[1],1)).T
Di = M - dfi.values
Di[Di<0] = 0
Di = Di[:,0:60]

dfo = pd.read_csv(home+"data/sims/covid.csv")
M = np.tile(df.Meta.values*100, (dfo.values.shape[1],1)).T
Do = M - dfo.values
Do[Do<0] = 0
Do = Do[:,0:60]

df5 = pd.read_csv(home+"data/sims/optimal_covid2.csv")
M = np.tile(df.Meta.values*100, (df5.values.shape[1],1)).T
D5 = M - df5.values
D5[D5<0] = 0
D5 = D5[:]












plt.figure(figsize=(6,4))
plt.plot( np.mean(Do*np.tile(weights, (Do.shape[1],1)).T, axis=0), '-k', linewidth=2, label='escenario pandemia' )
plt.plot( np.mean(D5*np.tile(weights, (D5.shape[1],1)).T, axis=0), '--k', linewidth=2, label='optimización bajo pandemia' )

plt.xlim(-1, Di.shape[1])
plt.gca().set_xticks(range(0, Di.shape[1]+1, 6))
plt.gca().set_xticklabels(range(2020, 2031))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('año', fontsize=14)
plt.ylabel('brecha ponderada promedio', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(home+'/figuras/covid_efecto2.pdf')
plt.show()












