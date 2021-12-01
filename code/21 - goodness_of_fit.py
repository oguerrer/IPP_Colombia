'''

Genera figuras sobre la bondad de ajuste de los parámetros.

'''

import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]


latexpath = home+'/figuras/'


os.chdir(home+'/code/')
import ppi
from functions import *



parallel_processes = 20
sample_size = 1000


df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")

colYears = [col for col in df.columns if str(col).isnumeric()]







dfc = pd.read_csv(home+"data/modeling/parameters.csv")
gof_a = dfc.GoF_alpha.values.tolist()
gof_b = dfc.GoF_beta.values.tolist()
sdgs = df.ODS.values.tolist()
indis = df.Abreviatura.values.tolist()


gof_ar = np.round(100*np.array(gof_a), 2)
gof_br = np.round(100*np.array(gof_b), 2)

x1, x2, y, label, sdg = [], [], [], [], []
for i in range(len(gof_ar)):
    x1.append( gof_ar[i] )
    x2.append( gof_br[i] )
    label.append( indis[i] )
    sdg.append( sdgs[i] )


dff = pd.DataFrame(np.array([label, sdg, x1, x2]).T, columns=['seriesCode', 'sdg', 'gof_alpha', 'gof_beta'])
dff = dff.astype({'sdg':int, 'gof_alpha':float, 'gof_beta':float})



file = open(home+"/data/misc/sdg_colors.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()






dff.sort_values(by=['sdg'], inplace=True)
ugof = np.unique(np.round(dff.gof_alpha.values, 0))

fig = plt.figure(figsize=(6,4))
heights = dict(zip(ugof, np.zeros(len(ugof))))
for index, row in dff.iterrows():
    x = np.round(row.gof_alpha, 0)
    y = heights[x]
    heights[x] += 1
    if df.loc[index].Instrumental==0:
        plt.plot(x, y, '.', mec='w', mfc=colors_sdg[row.sdg], markersize=16, markeredgewidth=.2)
    else: 
        plt.plot(x, y, 'p', mec='w', mfc=colors_sdg[row.sdg], markersize=10, markeredgewidth=.2)

# plt.xlim(90, 100.5)
plt.xlabel('bondad de ajuste', fontsize=14)
plt.ylabel('frecuencia', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().get_yaxis().grid()
plt.tight_layout()
plt.savefig(latexpath+'gof_alpha_sdg.pdf')
plt.show()








dff.sort_values(by=['sdg'], inplace=True)
ugof = np.unique(np.round(dff.gof_beta.values, 0))

fig = plt.figure(figsize=(6,4))
heights = dict(zip(ugof, np.zeros(len(ugof))))
for index, row in dff.iterrows():
    x = np.round(row.gof_beta, 0)
    y = heights[x]
    heights[x] += 1
    if df.loc[index].Instrumental==0:
        plt.plot(x, y, '.', mec='w', mfc=colors_sdg[row.sdg], markersize=16, markeredgewidth=.2)
    else: 
        plt.plot(x, y, 'p', mec='w', mfc=colors_sdg[row.sdg], markersize=10, markeredgewidth=.2)

# plt.xlim(90, 100.5)
plt.xlabel('bondad de ajuste', fontsize=14)
plt.ylabel('frecuencia', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().get_yaxis().grid()
plt.tight_layout()
plt.savefig(latexpath+'gof_beta_sdg.pdf')
plt.show()













