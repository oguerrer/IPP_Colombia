'''
Este script genera los cuadros 1 y C.1, así como la figura 1.

'''

import matplotlib.pyplot as plt
import numpy as np
import os, copy, re, csv
import pandas as pd


home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/base_final.csv")
colYears = [c for c in df.columns if c.isnumeric()]

file = open(home+"/data/color_codes.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()



all_sdgs = set()
tabla = []
# tabla indis nombres
for index, row in df.iterrows():
    ods1 = int(row.ODS1)
    all_sdgs.add(ods1)
    if np.isnan(row.ODS2):
        ods = str(ods1)
    else:
        ods = str(ods1) + '\\&' + str(int(row.ODS2))
        all_sdgs.add(row.ODS2)
    val = np.round(np.mean(row[colYears]), 3)
    std = np.round(np.std(row[colYears]), 3)
    nombre = row.Nombre.replace('  ', ' ').replace('%', '\\%')
    if nombre[-1] == '.':
        nombre = nombre[0:-1]
    nombre = re.sub(r" ?\([^)]+\)", "", nombre)
    isinst = 'sí'
    if row.Instrumental==0:
        isinst = 'no'
    
    tabla.append( [row.Abreviatura, nombre, ods,  isinst, val, std] )
    
dff = pd.DataFrame(tabla, columns=['abreviatura', 'nombre', 'ods',  'instrumental', 'media', 'desv_est'])
dff.to_csv(home+"/cuadros/indicadores.csv", index=False)







# tabla indis fuentes
tabla = []
for index, row in df.iterrows():
    ods1 = int(row.ODS1)
    all_sdgs.add(ods1)
    if np.isnan(row.ODS2):
        ods = str(ods1)
    else:
        ods = str(ods1) + '\\&' + str(int(row.ODS2))
        all_sdgs.add(row.ODS2)
    val = np.round(np.mean(row[colYears]), 3)
    std = np.round(np.std(row[colYears]), 3)
    nombre = row.Fuente.replace('  ', ' ').replace('%', '\\%')
    if nombre[-1] == '.':
        nombre = nombre[0:-1]
    nombre = re.sub(r" ?\([^)]+\)", "", nombre)
    isinst = 'sí'
    if row.Instrumental==0:
        isinst = 'no'
    
    tabla.append( [row.Abreviatura, nombre] )
    
dff = pd.DataFrame(tabla, columns=['abreviatura', 'fuente'])
dff.to_csv(home+"/cuadros/fuentes.csv", index=False)









plt.figure(figsize=(12,3))
i = 0
labels = []
for index, row in df.iterrows():
    if index < 64:
        ODS1 = row.ODS1
        ODS2 = row.ODS2
        meta = row.Meta
        plt.bar(i, row[colYears].values.mean(), color=colors_sdg[ODS1], width=.65)
        if ~np.isnan(ODS2):
            plt.bar(i, row[colYears].values.mean()/2, bottom=row[colYears].values.mean()/2, color=colors_sdg[ODS2], width=.65)
        plt.plot(i, meta, '.', markersize=15, mfc=colors_sdg[ODS1], mec='w')
        if row.Instrumental == 1:
            plt.plot(i, 0, '^k', markersize=10)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(0, 1)
plt.ylabel('nivel del\nindicador y meta', fontsize=14)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/niveles_1.pdf')
plt.show()






plt.figure(figsize=(12,3))
i = 0
labels = []
for index, row in df.iterrows():
    if index >= 64:
        ODS1 = row.ODS1
        ODS2 = row.ODS2
        meta = row.Meta
        plt.bar(i, row[colYears].values.mean(), color=colors_sdg[ODS1], width=.65)
        if ~np.isnan(ODS2):
            plt.bar(i, row[colYears].values.mean()/2, bottom=row[colYears].values.mean()/2, color=colors_sdg[ODS2], width=.65)
        plt.plot(i, meta, '.', markersize=15, mfc=colors_sdg[ODS1], mec='w')
        if row.Instrumental == 1:
            plt.plot(i, 0, '^k', markersize=10)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(0, 1)
plt.ylabel('nivel del\nindicador y meta', fontsize=14)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+'/figuras/niveles_2.pdf')
plt.show()

















































