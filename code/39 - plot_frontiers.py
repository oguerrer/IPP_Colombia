'''

Grafica resultados del análisis de frontera.

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





# Parameters
df_params = pd.read_csv(home+"data/modeling/parameters.csv")
alphas = df_params.alpha.values.copy()
betas = df_params.beta.values.copy()
T = int(df_params['T'].values[0])
scalar = df_params.scalar.values[0]
min_value = df_params.min_value.values[0]
num_years = df_params.years.values[0]
sub_periods = int(T/num_years)








dft = pd.read_csv(home+"data/sims/frontier.csv")
all_vals_t = dft.values[:,1::]

on_time=[]
late=[]
unfeasible=[]
feasible=[]
i=0
for index, row in df.iterrows():
    meta = scalar*row.Meta2030
    gaps = meta - all_vals_t[i]
    reaches = np.where(gaps<=0)[0]
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
plt.savefig(path+'dona_convergencia_frontera.pdf')
plt.show()








data_out = []

dft = pd.read_csv(home+"data/sims/frontier.csv")
all_vals_t = dft.values[:,1::]

dfp = pd.read_csv(home+"data/sims/prospective.csv")
all_vals_p = dfp.values[:,1::]

new_rows = []
plt.figure(figsize=(12,4))
plt.fill_between([-1, 50], [-10, -10], [10*12/2, 10*12/2], color='grey', alpha=.25)
for i, row in df.iterrows():
    
    ods = row.ODS
    pos = np.where(all_vals_t[i] >= all_vals_p[i][sub_periods*10])[0][0]
    savings = 12*(sub_periods*10-pos)/sub_periods
    fin_val = row[colYears].values.mean()*scalar

    new_rows.append([savings, row.Abreviatura, row.ODS, row.Instrumental, fin_val])
    data_out.append([row.Abreviatura, savings])

    if row.Instrumental == 0:
        plt.plot(fin_val, savings, '.', mec='w', mfc=colors_sdg[ods], markersize=25)
        plt.text(fin_val-0, savings+0, row.Abreviatura, fontsize=5, rotation=0, 
                 horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', 
                alpha=0.4, edgecolor='w', pad=0))
    else:
        plt.plot(fin_val, savings, 'p', mec='w', mfc=colors_sdg[ods], markersize=15)
        plt.text(fin_val-0, savings+0, row.Abreviatura, fontsize=5, rotation=0, 
                 horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', 
                alpha=0.4, edgecolor='w', pad=0))
    plt.text(12, 2, 'potenciales cuellos de botella estructurales', fontsize=12, 
             horizontalalignment='left', verticalalignment='center')

plt.xlim(10, 75)
plt.ylim(-5, 120)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.xlabel('nivel promedio del indicador (más=mejor)', fontsize=14)
plt.ylabel('meses de ahorro', fontsize=14)
plt.tight_layout()
plt.savefig(path+'frontier_months_2030.pdf')
plt.show()










new_rows2 = []
for meses, nombre, sdg, instr0, level in sorted(new_rows)[::-1]:
    
    instr = 'sí'
    if instr0 == 0:
        instr = 'no' 
    
    new_rows2.append([nombre, sdg, instr, meses, np.round(level,2)])


dfi = pd.DataFrame(new_rows2, columns=['Indicador', 'ODS', 'Instrumental', 'Meses de ahorro', 'Nivel promedio histórico'])
dfi.to_excel(home+'/cuadros/cuadro_B.1.xls')































































