'''
Este script genera las figuras B.1, B.2 y el panel a de la figura 8.

'''

import matplotlib.pyplot as plt
import numpy as np
import os, copy, re, csv
import pandas as pd
from scipy.optimize import curve_fit

home =  os.getcwd()[:-4]




df = pd.read_csv(home+"data/base_final.csv")
colYears = np.array([c for c in df.columns if c.isnumeric()])

file = open(home+"/data/color_codes.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()





dfi = pd.read_csv(home+"data/sims/prospective.csv")
all_vals = dfi.values

gaps0 = 100*df.Meta.values - all_vals[:,59]
gaps0[gaps0<0] = 0

interests = np.array(list(range(-5, 10, 1)))
reductions = 9.5-np.array(list(range(-5, 10, 1)))[::-1]

M = np.zeros((len(df), len(list(range(-5, 10, 1)))))

i=0
for j in range(9, -6, -1):
    dft = pd.read_csv(home+"data/sims/decrement_"+str(j)+".csv")
    all_vals_t = dft.values
    gaps_t = 100*df.Meta.values - all_vals_t[:,59]
    
    M[:, i] = 100*(gaps_t - gaps0)/gaps0
    i+=1

M[M<0] = 0    


def func(x, a):
    y = a*x
    y -= y[0]
    return y


    



plt.figure(figsize=(6,4))
params = []
indices = []
plt.fill_between([9.5, 100], [-100, -100], [1000, 1000], color='grey', alpha=.25)
for index, serie in enumerate(M):
    if gaps0[index] > 0:
        row = df.loc[index]
        x = reductions
        y = serie
        popt, pcov = curve_fit(func, x, y, p0=[0])
        plt.plot(x, func(x, popt[0]), color=colors_sdg[row.ODS1])
        params.append(popt[0])
        indices.append(index)
plt.xlim(.5, len(x)-.5)
plt.ylim(-5, 200)
plt.xlabel('reducción de tasa crecimiento presupuestal (%)', fontsize=14)
plt.ylabel('brecha de 2030 ampliada (%)', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/retraso_curves.pdf')
plt.show()















plt.figure(figsize=(6,4))
for i, index in enumerate(indices):
        row = df.loc[index]
        param = params[i]
        level = row['2020']*100
        if row.Instrumental==1:
            plt.semilogy(level, param, '.', mfc=colors_sdg[row.ODS1], mec='w', markersize=20)
        else:
            plt.semilogy(level, param, 's', mfc=colors_sdg[row.ODS1], mec='w', markersize=8)
# plt.xlim(1, len(x))
# plt.ylim(-1, 10)
plt.xlabel('nivel del indicador en 2020', fontsize=14)
plt.ylabel('parámetro de sensibilidad', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/retraso_params.pdf')
plt.show()














sarg = np.argsort(params)
plt.figure(figsize=(12,3.5))
i=0
labels = []
for arg in sarg:
    row = df.loc[indices[arg]]
    param = params[arg]
    if param > 0:
        labels.append(row.Abreviatura)
        if i%2==0:
            plt.plot([i,i], [-10, param], '-', color='grey', linewidth=.5)
        else:
            plt.plot([i,i], [-10, param], '--', color='black', linewidth=.5)
        if row.Instrumental==1:
            plt.semilogy(i, param, '.', mfc=colors_sdg[row.ODS1], mec='w', markersize=15)
        else:
            plt.semilogy(i, param, 's', mfc=colors_sdg[row.ODS1], mec='w', markersize=7)
        i+=1
# plt.ylim(-1., 16)
plt.xlim(-1, len(labels))
plt.gca().set_xticks(range(len(labels)))
plt.gca().set_xticklabels(labels, rotation=90, fontsize=7)
plt.ylabel('parámetro de sensibilidad', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/retraso_indis.pdf')
plt.show()










dft = pd.read_csv(home+"data/sims/decrement_-5.csv")
all_vals_t = dft.values

on_time=[]
late=[]
unfeasible=[]
feasible=[]
i=0
for index, row in df.iterrows():
    meta = 100*row.Meta
    reaches = np.where(all_vals_t[i] >= meta)[0]
    if len(reaches) > 0 and reaches[0]/6 <= 10:
        on_time.append(index)
        feasible.append(index)
    elif len(reaches) > 0 and reaches[0]/6 <= 20:
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

cin = [colors_sdg[df.loc[c].ODS1] for c in on_time] + [colors_sdg[df.loc[c].ODS1] for c in late] + [colors_sdg[df.loc[c].ODS1] for c in unfeasible]
labels = [df.loc[c].Abreviatura for c in on_time] + [df.loc[c].Abreviatura for c in late] + [df.loc[c].Abreviatura for c in unfeasible]
pie2, _ = ax.pie(np.ones(len(df)), radius=1, colors=cin, labels=labels, rotatelabels=True, shadow=False, counterclock=False,
                 startangle=90, textprops=dict(va="center", ha='center', rotation_mode='anchor', fontsize=5), 
                 labeldistance=1.17)
plt.setp( pie2, width=width, edgecolor='none')
plt.tight_layout()
plt.savefig(home+'/figuras/dona_convergencia_mitad.pdf')
plt.show()









































































