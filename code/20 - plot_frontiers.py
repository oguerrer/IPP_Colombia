'''
Este script genera  el panel c de la figura 8, y la figura 9.
También se genera el cuadro D.4.

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







dft = pd.read_csv(home+"data/sims/frontier.csv")
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
plt.savefig(home+'/figuras/dona_convergencia_frontera.pdf')
plt.show()










dft = pd.read_csv(home+"data/sims/frontier.csv")
all_vals_t = dft.values

dfp = pd.read_csv(home+"data/sims/prospective.csv")
all_vals_p = dfp.values


plt.figure(figsize=(12,4))
plt.fill_between([-1, 60], [-10, -10], [10*12/2, 10*12/2], color='grey', alpha=.25)
for i, row in df.iterrows():
    
    ods = row.ODS1
    pos = np.where(all_vals_t[i] >= all_vals_p[i][59])[0][0]
    savings = 12*(59-pos)/6
    fin_val = row[colYears].values.mean()*100

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
    plt.text(7, -1, 'potenciales cuellos de botella estructurales', fontsize=12, 
             horizontalalignment='left', verticalalignment='center')

plt.xlim(5, 95)
plt.ylim(-5, 120)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('nivel promedio del indicador (más=mejor)', fontsize=14)
plt.ylabel('meses de ahorro', fontsize=14)
plt.tight_layout()
plt.savefig(home+'/figuras/frontier_months_2030.pdf')
plt.show()












R = df.Instrumental.values==1
dfpm = dfp.values[:,59]
dffm = dft.values[:,0:60]
all_meses = []
for i in range(len(dfpm)):
    
    pos = np.where(dffm[i] >= dfpm[i])[0][0]+1
    savings = 12 * (60 - pos) / 6
    all_meses.append(savings)

args = np.argsort(all_meses)[::-1]
tabla = []
for arg in args:
    
    sdg = df.ODS1.values[arg]
    nombre = df.Abreviatura.values[arg]
    meses = all_meses[arg]
    level = np.round(dffm[arg][0], 2)
    instr = 'sí'
    if R[arg] == 0:
       instr = 'no' 
    
    tabla.append( [nombre, sdg, instr, meses, level] )

dff = pd.DataFrame(tabla, columns=['abreviatura', 'ods', 'instrumental', 'meses_ahorro', 'nivel_2020'])
dff.to_csv(home+'/cuadros/ahorros_bajo_fronteras.csv', index=False)
































































