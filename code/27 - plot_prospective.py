'''

Grafica los resultados de las corridas prospectivas.

'''

import matplotlib.pyplot as plt
import numpy as np
import os, copy, re, csv
import pandas as pd


home =  os.getcwd()[:-4]

path = home+'/figuras/'


df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = np.array([c for c in df.columns if c.isnumeric()])

file = open(home+"/data/misc/sdg_colors.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()




dfi = pd.read_csv(home+"data/sims/prospective.csv")
all_vals = dfi.values[:,1::]




# Parameters
df_params = pd.read_csv(home+"data/modeling/parameters.csv")
alphas = df_params.alpha.values.copy()
betas = df_params.beta.values.copy()
T = int(df_params['T'].values[0])
scalar = df_params.scalar.values[0]
min_value = df_params.min_value.values[0]
num_years = df_params.years.values[0]
sub_periods = int(T/num_years)



data_out = []

plt.figure(figsize=(6,4))
i = 0
labels = []
plt.fill_between([-10, 1000], [21, 21], [30, 30], color="grey", alpha=.25)
on_time = []
late = []
unfeasible = []
feasible = []
for index, row in df.iterrows():
    ODS1 = row.ODS
    meta = 100*row.Meta2030
    reaches = np.where(all_vals[i] >= meta)[0]
    
    if len(reaches) > 0:
        data_out.append([row.Abreviatura, reaches[0]/sub_periods, 100*row['2020']])
    else:
        data_out.append([row.Abreviatura, 'más de 20', 100*row['2020']])

    if len(reaches) > 0:
        if row.Instrumental==0:
            plt.plot(row['2020']*100, reaches[0]/sub_periods, '.', mfc=colors_sdg[ODS1], mec='w', markersize=16)
        else:
            plt.plot(row['2020']*100, reaches[0]/sub_periods, 'p', mfc=colors_sdg[ODS1], mec='w', markersize=10)
    else:
        if row.Instrumental==0:
            plt.plot(row['2020']*100, 22+1.5*np.random.rand(), '.', mfc=colors_sdg[ODS1], mec='w', markersize=16)
        else:
            plt.plot(row['2020']*100, 22+1.5*np.random.rand(), 'p', mfc=colors_sdg[ODS1], mec='w', markersize=10)
    
    if len(reaches) > 0 and reaches[0]/sub_periods <= 10:
        on_time.append(index)
        feasible.append(index)
    elif len(reaches) > 0 and reaches[0]/sub_periods <= 20:
        late.append(index)
        feasible.append(index)
    else:
        unfeasible.append(index)
    
    labels.append(row.Abreviatura)
    i+=1
plt.xlim(5, 95)
plt.ylim(-1, 25)
plt.ylabel('años para alcanzar la meta', fontsize=14)
plt.xlabel('nivel del indicador en 2020', fontsize=14)
plt.gca().set_yticks(list(range(0, 21, 5)) + [23])
plt.gca().set_yticklabels(list(range(0, 21, 5)) + ['>20'], rotation=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(path+'convergencia.pdf')
plt.show()


 










dff = df.loc[feasible]
all_vals_ff = all_vals[df.index.isin(feasible)]
convs = [np.where(all_vals_ff[i] >= 100*items[1].Meta2030)[0][0]/sub_periods for i, items in enumerate(dff.iterrows())]
sarg = np.argsort(convs)
labels = []

fig = plt.figure(figsize=(6,4))
for i, arg in enumerate(sarg):
    row = dff.loc[feasible[arg]]
    ODS1 = row.ODS
    labels.append(row.Abreviatura)
    if i%2!=0:
        plt.plot([i,i], [-10, convs[arg]], '--k', linewidth=.5)
    else:
        plt.plot([i,i], [-10, convs[arg]], '-', color='grey', linewidth=.5)
    if row.Instrumental==0:
        plt.plot(i, convs[arg], '.', mfc=colors_sdg[ODS1], mec='w', markersize=16)
    else:
        plt.plot(i, convs[arg], 'p', mfc=colors_sdg[ODS1], mec='w', markersize=10)
        
plt.xlim(-1, len(labels))
plt.ylim(-1, 21)
plt.gca().set_yticks(range(0, 21, 5))
plt.gca().set_yticklabels(['2021', '2025', '2030', '2035', '2040'])
plt.gca().set_xticks(range(len(labels)))
plt.gca().set_xticklabels(labels, rotation=90, fontsize=7)
plt.ylabel('fecha de convergencia', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(path+'convergencia_curva.pdf')
plt.show()









fig = plt.figure(figsize=(6,4))
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
plt.savefig(path+'dona_convergencia.pdf')
plt.show()











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
plt.savefig(path+'dona_convergencia_cuadrada.pdf')
plt.show()





















































