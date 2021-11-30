'''
Este script genera la base de datos presupuestal que usa IPP, así como
los paneles de la figura 2.

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


dfl = pd.read_excel(home+"data/raw/presupuesto_conjunto.xlsx", sheet_name='lima')
dfp = pd.read_excel(home+"data/raw/presupuesto_conjunto.xlsx", sheet_name='peru')




for index, row in dfl.iterrows():
    nulls = dfl.columns[row.isnull()].values.astype(int)
    if len(nulls) > 0:
        dfl.loc[index, nulls] = row[colYears.astype(int)].mean()



dfl['ODS'] = [meta.split('.')[0] for meta in dfl.Meta.values]

dfls = dfl.groupby('ODS').sum()


new_rows = []
for ods in sorted(df.ODS1.unique()):
    if str(ods) in dfls.index:
        line1 = [ods, 'Lima'] + (dfls[dfls.index==str(ods)][colYears.astype(int)].values).tolist()[0]
        new_rows.append(line1)
    if ods in dfp.ODS.values:
        line2 = [ods, 'Perú'] + (dfp[dfp.ODS==ods][colYears.astype(int)].values).tolist()[0]
        new_rows.append(line2)

dff = pd.DataFrame(new_rows, columns=['ODS', 'Entidad']+colYears.tolist())




for index, row in dff.iterrows():
    if index > 0:
        if row.ODS == dff.loc[index-1, 'ODS']:
            ratios = dff.loc[index-1, colYears].values/row[colYears].values
            plt.plot( ratios, color=colors_sdg[row.ODS] )
            if np.sum(ratios > 1):
                print(row.ODS)
plt.ylabel('presupuesto Lima/Perú')
plt.show()



dff.to_csv(home+'data/base_presupuesto.csv', index=False)












dfi = dff.groupby('ODS').sum()





plt.figure(figsize=(6,4))
plt.stackplot(colYears.astype(int), dfi.values/1000000, colors=[colors_sdg[c] for c in dfi.index])
plt.xlim(2012, 2020)
plt.ylabel('millones de soles reales', fontsize=14)
plt.xlabel('año', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/presupuesto_evol.pdf')
plt.show()







dft = dff[dff.Entidad=='Lima']
plt.figure(figsize=(6,4))
plt.stackplot(colYears.astype(int), dft[colYears].values/1000000, colors=[colors_sdg[c] for c in dft.ODS])
plt.xlim(2012, 2020)
plt.ylabel('millones de soles reales', fontsize=14)
plt.xlabel('año', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/presupuesto_evol_lima.pdf')
plt.show()







plt.figure(figsize=(6,4))
annual_change = dict(zip(dfi.index, np.mean(100*(dfi.values[:,1::] - dfi.values[:,0:-1])/dfi.values[:,0:-1], axis=1)))
for index, row in df.iterrows():
    vals = row[colYears].values
    indi_change = np.mean(100*((vals[1::] - vals[0:-1])/vals[0:-1]))
    ODS1 = row.ODS1
    ODS2 = row.ODS2
    if ODS1 in annual_change:
        plt.plot(annual_change[ODS1], indi_change, '.', mfc=colors_sdg[ODS1], mec='w', markersize=15)
    # if ODS2 in annual_change:
    #     plt.plot(annual_change[ODS2], indi_change, '.', mfc=colors_sdg[ODS2], mec='w', markersize=15)
# plt.xlim(2012, 2020)
plt.ylabel('cambio promedio del indicador', fontsize=14)
plt.xlabel('cambio promedio presupuestal del ODS', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/presupuesto_indis.pdf')
plt.show()









plt.figure(figsize=(6,4))
for index, row in df.iterrows():
    vals = row[colYears].values
    indi_level = np.mean(100*vals)
    indi_change = np.mean(100*((vals[1::] - vals[0:-1])/vals[0:-1]))
    ODS1 = row.ODS1
    ODS2 = row.ODS2
    plt.plot(indi_level, indi_change, '.', mfc=colors_sdg[ODS1], mec='w', markersize=15)
    # if str(ODS2) != 'nan':
    #     plt.plot(indi_level, indi_change, '.', mfc=colors_sdg[ODS2], mec='w', markersize=15)
# plt.xlim(2012, 2020)
plt.ylabel('cambio anual promedio', fontsize=14)
plt.xlabel('nivel anual promedio', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/indis_desempenio.pdf')
plt.show()





















































































