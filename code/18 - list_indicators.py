import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]

latexpath = home+'/figuras/'


file = open(home+"/data/misc/sdg_colors.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()


df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = [col for col in df.columns if str(col).isnumeric() if int(col)]



new_rows = []
for index, row in df.iterrows():
    inst = 'sí'
    if row.Instrumental==0:
        inst = 'no'
    
    new_rows.append([row.Abreviatura, row.Indicador, row.MetaODS,
          inst, '{0:.2f}'.format(np.mean(row[colYears])),
          '{0:.2f}'.format(np.std(row[colYears]))])

dfi = pd.DataFrame(new_rows, columns=['Código', 'Nombre', 'Target', 'Instrumental', 'Media', 'Desviación estándar'])
dfi.to_excel(home+'/cuadros/cuadro_1.xls')






plt.figure(figsize=(12,3))
i = 0
labels = []
for index, row in df.iterrows():
    if index < 49:
        ODS1 = row.ODS
        meta = row.Meta2030
        plt.bar(i, 100*row[colYears].values.mean(), color=colors_sdg[ODS1], width=.65)
        plt.plot(i, 100*meta, '.', markersize=15, mfc=colors_sdg[ODS1], mec='w')
        if row.Instrumental == 1:
            plt.plot(i, 0, '^k', markersize=10)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(0, 110)
plt.ylabel('nivel del\nindicador y meta', fontsize=14)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.savefig(latexpath+'niveles_1.pdf')
plt.show()










plt.figure(figsize=(12,3))
i = 0
labels = []
for index, row in df.iterrows():
    if index >= 49:
        ODS1 = row.ODS
        meta = row.Meta2030
        plt.bar(i, 100*row[colYears].values.mean(), color=colors_sdg[ODS1], width=.65)
        plt.plot(i, 100*meta, '.', markersize=15, mfc=colors_sdg[ODS1], mec='w')
        if row.Instrumental == 1:
            plt.plot(i, 0, '^k', markersize=10)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(0, 110)
plt.ylabel('nivel del\nindicador y meta', fontsize=14)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.savefig(latexpath+'niveles_2.pdf')
plt.show()















dfr = pd.read_csv(home+"data/preprocessed/indicators_reshaped.csv")
dfr.sort_values(by=['ODS', 'MetaODS'], inplace=True)
dfr.reset_index(inplace=True)

data_out = []

plt.figure(figsize=(12,3))
i = 0
labels = []
for index, row in dfr.iterrows():
    if index < 49:
        ODS1 = row.ODS
        meta = row.Meta2030
        value = row[colYears].dropna().values[-1]
        if row.Invertir==1 and meta<value:
            plt.bar(i, 100*meta/value, color=colors_sdg[ODS1], width=.65)
            data_out.append([row.Abreviatura, 100*meta/value])
        elif row.Invertir==1 and meta>value:
            plt.bar(i, 100*1, color=colors_sdg[ODS1], width=.65)
            data_out.append([row.Abreviatura, 100*1])
        else:
            plt.bar(i, 100*value/meta, color=colors_sdg[ODS1], width=.65)
            data_out.append([row.Abreviatura, 100*value/meta])
        if df.loc[index].Instrumental == 1:
            plt.plot(i, 0, '^k', markersize=10)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(0, 100)
plt.ylabel('tasa nivel-meta', fontsize=14)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.savefig(latexpath+'metas_1.pdf')
plt.show()










plt.figure(figsize=(12,3))
i = 0
labels = []
for index, row in dfr.iterrows():
    if index >= 49:
        ODS1 = row.ODS
        meta = row.Meta2030
        value = row[colYears].dropna().values[-1]
        if row.Abreviatura=='victi_homi':
            plt.bar(i, 200, color='w', hatch='/////', width=.65)
            data_out.append([row.Abreviatura, ''])
        else:
            if row.Invertir==1 and meta<value:
                plt.bar(i, 100*meta/value, color=colors_sdg[ODS1], width=.65)
                data_out.append([row.Abreviatura, 100*meta/value])
            elif row.Invertir==1 and meta>value:
                plt.bar(i, 100*1, color=colors_sdg[ODS1], width=.65)
                data_out.append([row.Abreviatura, 100*1])
            else:
                plt.bar(i, 100*value/meta, color=colors_sdg[ODS1], width=.65)
                data_out.append([row.Abreviatura, 100*value/meta])
            if df.loc[index].Instrumental == 1:
                plt.plot(i, 0, '^k', markersize=10)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(0, 100)
plt.ylabel('tasa nivel-meta', fontsize=14)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.savefig(latexpath+'metas_2.pdf')
plt.show()














plt.figure(figsize=(12,3))
i = 0
labels = []
for index, row in dfr.iterrows():
    if index < 49:
        ODS1 = row.ODS
        meta = row.Meta2030
        value_fin = row[colYears].dropna().values[-1]
        value_ini = row[colYears].dropna().values[0]
        if row.Invertir==1:
            plt.bar(i, -100*(value_fin-value_ini)/value_ini, color=colors_sdg[ODS1], width=.65)
        else:
            plt.bar(i, 100*(value_fin-value_ini)/value_ini, color=colors_sdg[ODS1], width=.65)
        if df.loc[index].Instrumental == 1:
            plt.plot(i, 0, '^k', markersize=10)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(-100, 100)
plt.ylabel('tasa de avance', fontsize=14)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.savefig(latexpath+'avances_1.pdf')
plt.show()






plt.figure(figsize=(12,3))
i = 0
labels = []
for index, row in dfr.iterrows():
    if index >= 49:
        ODS1 = row.ODS
        meta = row.Meta2030
        value_fin = row[colYears].dropna().values[-1]
        value_ini = row[colYears].dropna().values[0]
        if row.Invertir==1:
            plt.bar(i, -100*(value_fin-value_ini)/value_ini, color=colors_sdg[ODS1], width=.65)
        else:
            plt.bar(i, 100*(value_fin-value_ini)/value_ini, color=colors_sdg[ODS1], width=.65)
        if df.loc[index].Instrumental == 1:
            plt.plot(i, 0, '^k', markersize=10)
        labels.append(row.Abreviatura)
        i+=1
plt.xlim(-1, i)
plt.ylim(-100, 100)
plt.ylabel('tasa de avance', fontsize=14)
plt.gca().set_xticks(range(i))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.savefig(latexpath+'avances_2.pdf')
plt.show()







































