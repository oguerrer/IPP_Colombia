'''

Toma los datos presupuestales originales y los agrega a nivel target.
También genera las figuras del presupuesto usadas en el reporte.

'''

import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
import csv
import scipy.stats as st
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]


latexpath = home+'figuras/'


df_indis = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = [col for col in df_indis.columns if str(col).isnumeric() and int(col)]



file = open(home+"/data/misc/sdg_colors.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()


# Datos poblacionales
dfp = pd.read_excel(home+"data/raw/API_SP.POP.TOTL_DS2_en_excel_v2_2764317.xls", sheet_name='Data', skiprows=3)
colYears2 = [col for col in df_indis.columns if str(col).isnumeric() and int(col)]
total_pop = dfp[dfp['Country Code']=='COL'][colYears2].values[0]


# Cargar datos presupuestales vinculados a los ODS
df = pd.read_excel(home+"data/raw/07.09.2021_SIIF 2020 FINAL IPP.xlsx", sheet_name='SIIF 2020', skiprows=1)
budgets_targets = {}
for target in range(1, 171):
    budgets_targets[target] = df[str(target)].sum()



# Agregar datos a nivel target
df_meta = pd.read_excel(home+"data/raw/07.09.2021_SIIF 2020 FINAL IPP.xlsx", sheet_name='EQUIVALENCIAS METAS', skiprows=2)
metas_dict = dict(zip(df_meta['No Meta ODS'], df_meta['Número Meta Ejercicio']))


budgets_targets_final = {}
for target in df_indis.MetaODS:
    budgets_targets_final[target] = budgets_targets[metas_dict[target]]
for key, value in budgets_targets_final.items():
    budgets_targets_final[key] = budgets_targets_final[key]


budgets_sdgs_final = dict([(sdg, 0) for sdg in range(1,18)])
for key, value in budgets_targets_final.items():
    budgets_sdgs_final[int(key.split('.')[0])] += value
for key, value in budgets_sdgs_final.items():
    budgets_sdgs_final[key] = budgets_sdgs_final[key]


budgets_targets_final = dict([(key, val/total_pop[-1]) for key, val in budgets_targets_final.items() if key in df_indis.MetaODS.values])
for target in df_indis[~df_indis.MetaODS.isin(budgets_targets_final)].MetaODS.values:
    budgets_targets_final[target] = 0
zeros = [c for c in budgets_targets_final.keys() if budgets_targets_final[c]==0]
for zero in zeros:
    budgets_targets_final.pop(zero)



budgets_targets_final = dict([(target, bud) for target, bud in budgets_targets_final.items() if df_indis[df_indis.MetaODS==target].Instrumental.sum()>0])
stargets = sorted(budgets_targets_final.keys(), key=lambda x: (int(x.split('.')[0]), x.split('.')[1]) )

dft = pd.DataFrame([[c, budgets_targets_final[c]] for c in stargets], columns=['MetaODS', 'gasto'])
dft.to_csv(home+"data/modeling/budget_last_targets.csv", index=False)
  



df_indis.loc[:, 'Instrumental'] = [row.Instrumental if row.MetaODS in budgets_targets_final else 0 for index, row in df_indis.iterrows()]
df_indis.sort_values(by=['ODS', 'MetaODS'], inplace=True)
df_indis.to_csv(home+"data/modeling/indicators_sample_final.csv", index=False)






# Usar datos históricos presupuestales
dfg = pd.read_excel(home+"data/raw/Colombia_PresupuestoGral_2000-2021 SIIF_Feb2021.xlsx", sheet_name='PGN 2000 - 2020')
dfgs = dfg.groupby('AÑO').sum()
gasto = dict(zip(dfgs.index, dfgs.Compromisos))


dfx = pd.read_excel(home+"data/raw/1.2.5.IPC_Serie_variaciones_Jul1954-Feb2021.xlsx", sheet_name='Sheet1', skiprows=8)
dfx.loc[:, 'anio'] = [str(c)[0:4] for c in dfx['Año(aaaa)-Mes(mm)']]
dfxm = dfx.groupby('anio').mean()
ipc = dict(zip(dfxm.index, dfxm['Índice de Precios al Consumidor (IPC)']))


total_exp = np.array([ gasto[int(year)]*( ipc['2020']/ipc[year] )/total_pop[i] for i, year in enumerate(colYears) ])
pd.DataFrame([total_exp], columns=colYears).to_csv(home+"data/preprocessed/gasto_total.csv", index=False)
ref_val = total_exp[-1]



# Proyectar datos presupuestales hacia atras
M = np.zeros((len(budgets_targets_final), len(colYears)))
for i, year in enumerate(colYears):
    M[:,i] = [budgets_targets_final[target]*(total_exp[i]/ref_val) for target in stargets]
M = np.hstack( ([[c] for c in stargets], M) )
dff = pd.DataFrame(M, columns=['MetaODS'] + colYears)
dff.to_csv(home+"data/modeling/budget_target.csv", index=False)






def set_percentage(x):
    if x<3:
        return None
    else:
        return '{:0.0f}'.format(x)+'%'






fig = plt.figure(figsize=(6,4))
labels, sizes = zip(*budgets_targets_final.items())
percs = np.array(sizes)/np.sum(sizes)
labels2 = np.array(labels)
labels2[percs<.03] = ''
colors = [colors_sdg[int(target.split('.')[0])] for target in labels]
explode = [.04 for sdg in labels]
plt.pie(sizes, explode=explode, colors=colors, counterclock=False, pctdistance=.8,
        autopct=set_percentage, shadow=False, startangle=90, labels=labels2)
plt.tight_layout()
plt.savefig(latexpath+'/presupuesto_pay_target.pdf')
plt.show()











fig = plt.figure(figsize=(6,4))
plt.plot(colYears2, total_exp/1000000, '-k', linewidth=3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.xticks(colYears2[0::2])
plt.xlabel('año', fontsize=14)
plt.ylabel('millones de pesos per cápita', fontsize=14)
plt.tight_layout()
plt.savefig(latexpath+'/presupuesto_evol.pdf')
plt.show()















plt.figure(figsize=(6,4))
for index, row in df_indis.iterrows():
    vals = row[colYears].values
    indi_level = np.mean(100*vals)
    indi_change = np.mean(100*((vals[1::] - vals[0:-1])/vals[0:-1]))
    ODS1 = row.ODS
    if row.Instrumental==0:
        plt.plot(indi_level, indi_change, '.', mfc=colors_sdg[ODS1], mec='w', markersize=16)
    else:
        plt.plot(indi_level, indi_change, 'p', mfc=colors_sdg[ODS1], mec='w', markersize=10)
# plt.ylim(-10,50)
plt.ylabel('cambio anual promedio', fontsize=14)
plt.xlabel('nivel anual promedio', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(latexpath+'indis_desempenio.pdf')
plt.show()
















plt.figure(figsize=(6,4))
for index, row in df_indis.iterrows():
    vals = row[colYears].values
    indi_change = np.mean(100*((vals[1::] - vals[0:-1])/vals[0:-1]))
    target = row.MetaODS
    if target in budgets_targets_final:
        if row.Instrumental==0:
            plt.semilogx(budgets_targets_final[target], indi_change, '.', mfc=colors_sdg[int(target.split('.')[0])], mec='w', markersize=16)
        else:
            plt.semilogx(budgets_targets_final[target], indi_change, 'p', mfc=colors_sdg[int(target.split('.')[0])], mec='w', markersize=10)
# plt.ylim(-10,50)
plt.ylabel('cambio promedio del indicador', fontsize=14)
plt.xlabel('gasto a nivel target en 2020 (pesos per cápita)', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(latexpath+'presupuesto_indis_target.pdf')
plt.show()





















































































































































