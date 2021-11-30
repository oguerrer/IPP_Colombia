'''
Este script genera los paneles de las figuras 3 y 4, 
así como los cuadros D.1 y D.2.


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




dfi = pd.read_csv(home+"data/sims/prospective.csv")
all_vals = dfi.values




plt.figure(figsize=(6,4))
i = 0
labels = []
plt.fill_between([-10, 1000], [21, 21], [30, 30], color="grey", alpha=.25)
on_time = []
late = []
unfeasible = []
feasible = []
for index, row in df.iterrows():
    ODS1 = row.ODS1
    ODS2 = row.ODS2
    meta = 100*row.Meta
    reaches = np.where(all_vals[i] >= meta)[0]
    if len(reaches) > 0:
        plt.plot(row['2020']*100, reaches[0]/6, '.', mfc=colors_sdg[ODS1], mec='w', markersize=15)
    else:
        plt.plot(row['2020']*100, 22+1.5*np.random.rand(), '.', mfc=colors_sdg[ODS1], mec='w', markersize=15)
    
    if len(reaches) > 0 and reaches[0]/6 <= 10:
        on_time.append(index)
        feasible.append(index)
    elif len(reaches) > 0 and reaches[0]/6 <= 20:
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
plt.tight_layout()
plt.savefig(home+'/figuras/convergencia.pdf')
plt.show()


 



dfp = pd.read_csv(home+"data/sims/prospective_Ps.csv")
pro_ps = dfp.values

dfr = pd.read_csv(home+"data/sims/retrospective_Ps.csv")
ret_ps = dfr.values

changes_bud = 100*(pro_ps.mean(axis=1) - ret_ps.mean(axis=1))/ret_ps.mean(axis=1)

dfs = df[df.Instrumental==1]
all_vals_ins = all_vals[df.Instrumental==1]

fig = plt.figure(figsize=(6,4))
i=0
for index, row in dfs.iterrows():
    ODS1 = row.ODS1
    meta = 100*row.Meta
    reaches = np.where(all_vals_ins[i] >= meta)[0]
    gap0 = max([0, meta - all_vals_ins[i,0]])
    gapt = max([0, meta - all_vals_ins[i,59]])
    if gap0 > 0:
    # if len(reaches) > 0:
        # plt.semilogx(changes_bud[i], reaches[0]/6, '.', mfc=colors_sdg[ODS1], mec='w', markersize=15)
        plt.semilogx(changes_bud[i], 100*(gap0-gapt)/gap0, '.', mfc=colors_sdg[ODS1], mec='w', markersize=15)
    
    i+=1
plt.ylabel('cierre de brecha (%)', fontsize=14)
plt.xlabel('cambio porcentual en el gasto', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/convergencia_gasto.pdf')
plt.show()







dff = df.loc[feasible]
all_vals_ff = all_vals[df.index.isin(feasible)]
convs = [np.where(all_vals_ff[i] >= 100*items[1].Meta)[0][0]/6 for i, items in enumerate(dff.iterrows())]
sarg = np.argsort(convs)
labels = []

fig = plt.figure(figsize=(6,4))
for i, arg in enumerate(sarg):
    row = dff.loc[feasible[arg]]
    ODS1 = row.ODS1
    labels.append(row.Abreviatura)
    if i%2!=0:
        plt.plot([i,i], [-10, convs[arg]], '--k', linewidth=.5)
    else:
        plt.plot([i,i], [-10, convs[arg]], '-', color='grey', linewidth=.5)
    plt.plot(i, convs[arg], '.', mfc=colors_sdg[ODS1], mec='w', markersize=15)
plt.xlim(-1, len(labels))
plt.ylim(-1, 21)
plt.gca().set_yticks(range(0, 21, 5))
plt.gca().set_yticklabels(['2021', '2025', '2030', '2035', '2040'])
plt.gca().set_xticks(range(len(labels)))
plt.gca().set_xticklabels(labels, rotation=90, fontsize=7)
plt.ylabel('fecha de convergencia', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/convergencia_curva.pdf')
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

cin = [colors_sdg[df.loc[c].ODS1] for c in on_time] + [colors_sdg[df.loc[c].ODS1] for c in late] + [colors_sdg[df.loc[c].ODS1] for c in unfeasible]
labels = [df.loc[c].Abreviatura for c in on_time] + [df.loc[c].Abreviatura for c in late] + [df.loc[c].Abreviatura for c in unfeasible]
pie2, _ = ax.pie(np.ones(len(df)), radius=1, colors=cin, labels=labels, rotatelabels=True, shadow=False, counterclock=False,
                 startangle=90, textprops=dict(va="center", ha='center', rotation_mode='anchor', fontsize=5), 
                 labeldistance=1.17)
plt.setp( pie2, width=width, edgecolor='none')
plt.tight_layout()
plt.savefig(home+'/figuras/dona_convergencia.pdf')
plt.show()






# TABLAS




dfi = pd.read_csv(home+"data/sims/prospective.csv")


df_table = pd.read_excel(home+"data/raw/Indicadores priorizados MML y de acuerdo a PDC.xlsx", sheet_name='list2')
selected = df_table.seriesCode.values

tabla = []
for index, row in df.iterrows():
    if row.Abreviatura in selected:
        
        val2020 = dfi.iloc[index].values[0]
        val2030 = dfi.iloc[index].values[59]
        val2035 = dfi.iloc[index].values[89]
        
        change30 = '{:.3f}'.format(100*(val2030-val2020)/val2020)
        change35 = '{:.3f}'.format(100*(val2035-val2020)/val2020)
        
        tabla.append( [row.Abreviatura, row.ODS1, change30, change35] )

dff = pd.DataFrame(tabla, columns=['abreviatura', 'ods', 'mejora_en_2030', 'mejora_en_2035'])
dff.to_csv(home+'/cuadros/proyecciones_estrategicos.csv', index=False)










df_table = pd.read_excel(home+"data/raw/Indicadores priorizados MML y de acuerdo a PDC.xlsx", sheet_name='list3')
selected = df_table.seriesCode.values


tabla = []
for index, row in df.iterrows():
    if row.Abreviatura in selected:
        
        val2020 = dfi.iloc[index].values[0]
        val2030 = dfi.iloc[index].values[59]
        val2035 = dfi.iloc[index].values[89]
        
        change30 = '{:.3f}'.format(100*(val2030-val2020)/val2020)
        change35 = '{:.3f}'.format(100*(val2035-val2020)/val2020)
        
        tabla.append( [row.Abreviatura, row.ODS1, change30, change35] )

dff = pd.DataFrame(tabla, columns=['abreviatura', 'ods', 'mejora_en_2030', 'mejora_en_2035'])
dff.to_csv(home+'/cuadros/proyecciones_complementarios.csv', index=False)











































