'''
Este script genera las figuras 10 y 11.

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

R = df.Instrumental==1

# Initial factors
adf = pd.read_csv(home+"data/parametros.csv")
alphas = adf.alphas.values.copy()
max_steps = int(adf.steps.values[0])
betas = adf.beta.values.copy()
num_years = adf.years.values[0]
scalar = adf.scalar.values[0]
min_value = adf.min_value.values[0]


dfb0 = pd.read_csv(home+"data/optimal_budget.csv")
bdf0 = dfb0.values[-1,0:-1]/dfb0.values[-1,0:-1].sum()

dfb50 = pd.read_csv(home+"data/optimal_budget_50.csv")
bdf50 = dfb50.values[-1,0:-1]/dfb50.values[-1,0:-1].sum()

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
min_fracs50 = np.ones(len(dfb))*.5



bench = dfb['2020'].values/max_steps
bench = bench[np.in1d(dfb.index, df.ODS1.unique())]
B_seq0 = min_fracs0*bench + (bench.sum()-(min_fracs0*bench).sum())*bdf0
B_seq50 = min_fracs50*bench + (bench.sum()-(min_fracs50*bench).sum())*bdf50





# Tabla de rigideces

bd = dfb['2020'].values
for i, ods in enumerate(sorted(dfbo.ODS.unique())):
    budget_prop = '{:.2f}'.format(100*bd[i])
    frac0 = '{:.2f}'.format(100*(1-min_fracs0[i]))
    frac50 = '{:.2f}'.format(100*min_fracs50[i])
    print( ods, '&', budget_prop, '&', frac0, '&', frac50, '\\\\' )









interest = 0.095
dfb_pro = pd.read_csv(home+"data/base_presupuesto.csv")
dfb_pro2 = dfb_pro.groupby('ODS').sum()
bench = dfb_pro2['2020'].values/(max_steps/num_years)
B_sequence_ref = np.zeros((len(dfb_pro2), int(10*max_steps/num_years)))
for i in range(B_sequence_ref.shape[1]):
    B_sequence_ref[:,i] = bench
    if i%int(max_steps/num_years)==0:
        bench *= (1+interest)


interest = 0.2
dfb_opt = pd.read_csv(home+"data/optimal_budget.csv")
bdf = dfb_opt.values[-1,0:-1]/dfb_opt.values[-1,0:-1].sum()
bud_fracs = []
for index, row in dfb_pro.iterrows():
    if index > 0:
        if row.ODS == dfb_pro.loc[index-1].ODS:
            bud_fracs.append( np.min([1, np.mean(dfb_pro.loc[index-1][colYears].values/row[colYears].values)]) )
        elif row.ODS != dfb_pro.loc[index+1].ODS:
            bud_fracs.append(0)
halves = np.ones(len(bud_fracs))*.5
min_fracs = np.max([halves, 1-np.array(bud_fracs)], axis=0)
bench = dfb_pro2['2020'].values/(max_steps/num_years)
B_sequence = np.zeros((len(dfb_pro2), int(10*max_steps/num_years)))
for i in range(B_sequence.shape[1]):
    B_sequence[:,i] = min_fracs*bench + (bench.sum()-(min_fracs*bench).sum())*bdf
    if i%int(max_steps/num_years)==0:
            bench *= (1+interest)



changes = 100*(B_sequence.sum(axis=1)-B_sequence_ref.sum(axis=1))/B_sequence_ref.sum(axis=1)
sargll = np.argsort(np.abs(changes))
rankll = np.argsort(sargll)
n_sdgs = len(dfbo.ODS.unique())


plt.figure(figsize=(6,4))
for i, ods in enumerate(sorted(dfbo.ODS.unique())):
    plt.arrow( i, 0, 0, np.sign(changes[i])*(rankll[i]+1), width=.5, 
              head_width=.7, head_length=1.0, color=colors_sdg[ods] )
    text = '{:.1f}'.format(changes[i])+'%'
    align = 1.5
    if changes[i] < 0:
        align *= -1
    plt.text( i, np.sign(changes[i])*(rankll[i]+1)+align, text, ha='center', va='center')
plt.ylim(0, max(rankll)+5)
plt.gca().set_yticks([])
plt.gca().set_xticks(range(len(dfb)))
plt.gca().set_xticklabels(sorted(dfbo.ODS.unique()))
plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('ODS', fontsize=14)
plt.tight_layout()
plt.savefig(home+'/figuras/flechas_0.pdf')
plt.show()










interest = 0.2
dfb_opt = pd.read_csv(home+"data/optimal_budget_50.csv")
bdf = dfb_opt.values[-1,0:-1]/dfb_opt.values[-1,0:-1].sum()
bud_fracs = []
for index, row in dfb_pro.iterrows():
    if index > 0:
        if row.ODS == dfb_pro.loc[index-1].ODS:
            bud_fracs.append( np.min([1, np.mean(dfb_pro.loc[index-1][colYears].values/row[colYears].values)]) )
        elif row.ODS != dfb_pro.loc[index+1].ODS:
            bud_fracs.append(0)
halves = np.ones(len(bud_fracs))*.5
min_fracs = np.ones(len(dfb))*.5
bench = dfb_pro2['2020'].values/(max_steps/num_years)
B_sequence = np.zeros((len(dfb_pro2), int(10*max_steps/num_years)))
for i in range(B_sequence.shape[1]):
    B_sequence[:,i] = min_fracs*bench + (bench.sum()-(min_fracs*bench).sum())*bdf
    if i%int(max_steps/num_years)==0:
            bench *= (1+interest)


changes = 100*(B_sequence.sum(axis=1)-B_sequence_ref.sum(axis=1))/B_sequence_ref.sum(axis=1)
sargll = np.argsort(np.abs(changes))
rankll = np.argsort(sargll)
n_sdgs = len(dfbo.ODS.unique())

plt.figure(figsize=(6,4))
for i, ods in enumerate(sorted(dfbo.ODS.unique())):
    plt.arrow( i, 0, 0, np.sign(changes[i])*(rankll[i]+1), width=.5, 
              head_width=.7, head_length=1.0, color=colors_sdg[ods] )
    text = '{:.1f}'.format(changes[i])+'%'
    align = 1.5
    if changes[i] < 0:
        align *= -1
    plt.text( i, np.sign(changes[i])*(rankll[i]+1)+align, text, ha='center', va='center')
plt.ylim(-max(rankll)+5, max(rankll)+5)
plt.gca().set_yticks([])
plt.gca().set_xticks(range(len(dfb)))
plt.gca().set_xticklabels(sorted(dfbo.ODS.unique()))
plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('ODS', fontsize=14)
plt.tight_layout()
plt.savefig(home+'/figuras/flechas_50.pdf')
plt.show()









dfi = pd.read_csv(home+"data/sims/prospective.csv")
M = np.tile(df.Meta.values*100, (dfi.values.shape[1],1)).T
Di = M - dfi.values
Di[Di<0] = 0
Di = Di[:,0:60]

dfo = pd.read_csv(home+"data/sims/optimal_muni.csv")
M = np.tile(df.Meta.values*100, (dfo.values.shape[1],1)).T
Do = M - dfo.values
Do[Do<0] = 0
Do = Do[:]

df5 = pd.read_csv(home+"data/sims/optimal_50.csv")
M = np.tile(df.Meta.values*100, (df5.values.shape[1],1)).T
D5 = M - df5.values
D5[D5<0] = 0
D5 = D5[:]

dft = pd.read_csv(home+"data/sims/increment_20.csv")
M = np.tile(df.Meta.values*100, (dft.values.shape[1],1)).T
Dt = M - dft.values
Dt[Dt<0] = 0
Dt = Dt[:,0:60]











plt.figure(figsize=(6,4))
plt.plot( Di.mean(axis=0), '-k', linewidth=2, label='presupuesto de 2020' )
# plt.plot( range(0, 60, 6), Di.mean(axis=0)[0::6], 'vk', label='presupuesto de 2020' )
plt.plot( Dt.mean(axis=0), '--k', linewidth=2, label='presupuesto creciendo al 20%' )
# plt.plot( range(0, 60, 6), Dt.mean(axis=0)[0::6], '^k', label='presupuesto creciendo al 20%' )
plt.plot( Do.mean(axis=0), '-.k', linewidth=2, label='reasignación conservadora' )
# plt.plot( range(0, 60, 6), Do.mean(axis=0)[0::6], '.k', label='reasignación conservadora', markersize=15 )
plt.plot( D5.mean(axis=0), ':k', linewidth=2, label='reasignación flexible' )
# plt.plot( range(0, 60, 6), D5.mean(axis=0)[0::6], 'o', label='reasignación flexible', markersize=10, mfc='w', mec='k' )

plt.xlim(-1, Dt.shape[1])
plt.gca().set_xticks(range(0, Dt.shape[1]+1, 6))
plt.gca().set_xticklabels(range(2020, 2031))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('año', fontsize=14)
plt.ylabel('brecha promedio en 2030', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(home+'/figuras/optimo_efecto.pdf')
plt.show()










D0 = 100*(df.Meta.values - df['2020'].values)
D0[D0<0] = 0

cierre_i = 100*(1-Di[:,-1]/D0)
cierre_i[cierre_i<0] = 0 
cierre_t = 100*(1-Dt[:,-1]/D0)
cierre_t[cierre_t<0] = 0 
cierre_o = 100*(1-Do[:,-1]/D0)
cierre_o[cierre_o<0] = 0
cierre_5 = 100*(1-D5[:,-1]/D0)
cierre_5[cierre_5<0] = 0



plt.figure(figsize=(12,3.5))
i=0
labels=[]
for index, row in df.iterrows():
    if D0[index] != 0:
        if i%2!=0:
            plt.fill_between([i-.5, i+.5], [-1000, -1000], [1000, 1000], color='grey', alpha=.25)
        plt.plot(i, cierre_i[index], 'v', mfc=colors_sdg[row.ODS1], mec='w', markersize=6)
        plt.plot(i, cierre_t[index], '^', mfc=colors_sdg[row.ODS1], mec='w', markersize=6)
        plt.plot(i, cierre_o[index], '.', mfc=colors_sdg[row.ODS1], mec='w', markersize=10)
        plt.plot(i, cierre_5[index], 'o', mfc='none', mec=colors_sdg[row.ODS1], markersize=6)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(-5, 105)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90, fontsize=7)
plt.ylabel('cierre de brecha (%)', fontsize=14)
plt.tight_layout()
plt.savefig(home+'/figuras/optimo_final.pdf')
plt.show()







































































































