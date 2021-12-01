'''

Clasifica todos los indicadores instrumentales de acuerdo a los lineamientos
propuestos en el diagrama de flujo de la figura 16 del reporte.

La tabla con las clasificaciones es guardada y puede usarse para reconstruir los
cuadros 3 y 4 del reporte. 


'''

import matplotlib.pyplot as plt
import numpy as np
import os, csv
import pandas as pd
from joblib import Parallel, delayed

home =  os.getcwd()[:-4]

os.chdir(home+'/code/')
import ppi

 


df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = [col for col in df.columns if col.isnumeric()]



file = open(home+"data/misc/nombres_cortos.txt", 'r')
sdg_names = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()




# Initial factors
adf = pd.read_csv(home+"data/modeling/parameters.csv")
alphas = adf.alpha.values.copy()
max_steps = int(adf['T'].values[0])
betas = adf.beta.values.copy()
num_years = adf.years.values[0]
scalar = adf.scalar.values[0]
min_value = adf.min_value.values[0]
I0 = df['2020'].values.copy()*scalar
R = df['Instrumental'].values==1
periods = max_steps/num_years



dfpros = pd.read_csv(home+"data/sims/prospective.csv").values[:,1::]
dfgood = pd.read_csv(home+"data/sims/increment_"+str(20)+".csv")
dffront = pd.read_csv(home+"data/sims/frontier.csv").values[:,1::]
dfopt = pd.read_csv(home+"data/sims/optimal_prospective_0.csv")



new_rows = []
for index, row in df.iterrows():
    i = index
    if row.Instrumental == 1:
        
        print(row.Abreviatura)
        
        nivel_historico = row[colYears].values.mean()*scalar
        new_row = [row.Abreviatura, row.Indicador, row.ODS, row.MetaODS, nivel_historico]
        
        meta = row['Meta2030']*scalar
        val2020 = row['2020']*scalar
        gap2020 = 100*max([0, meta-val2020])/meta
        new_row.append(gap2020)
    
        serie = dfpros[i,::]
        val2030 = serie[39]
        gap2030 = 100*max([0, meta-val2030])/meta
        brecha_2030 = gap2030
        new_row.append(gap2030)
        serie_pros = serie.copy()
        
        serie = dfgood.values[index,:]
        val2030 = serie[39]
        gap2030 = 100*max([0, meta-val2030])/meta
        new_row.append(gap2030)
        
        serie = dffront[i,::]
        val2030 = serie[39]
        gap2030 = 100*max([0, meta-val2030])/meta
        new_row.append(gap2030)
        # pos_2030 = np.where(dffront[i] >= dfpros[i,39])[0][0]
        pos_2030 = np.argmin(np.abs(dffront[i]-dfpros[i,39]))
        ahorro = 12 * (39 - pos_2030) / periods
        
        
        serie = dfopt[dfopt.Abreviatura == row.Abreviatura].values[0][1::]
        val2030 = serie[39]
        gap2030 = 100*max([0, meta-val2030])/meta
        new_row.append(gap2030)
        new_row.append(ahorro)
        
        lineamiento = ''
        if brecha_2030 == 0:
            lineamiento = 'no hacer ajustes presupuestales'
        else:
            if ahorro > 60:
                lineamiento = 'ajustar presupuesto de acuerdo con el criterio de gasto óptimo'
            else:
                if nivel_historico < 50:
                    lineamiento = 'Revisar el funcionamiento de los programas de gobierno asociados en busca de cuellos de botella'
                else:
                    lineamiento = 'no hacer ajustes presupuestales'
        
        new_row.append(lineamiento)
    
        new_rows.append(new_row)


data_out = pd.DataFrame(new_rows, columns=['Indicador', 'Nombre', 'ODS', 'MetaODS', 'Nivel_promedio', 'Brecha_2020',
                                           'Brecha_2030', 
                                           'Brecha_2030_mas_20%',
                                           'Brecha_2030_frontera%',
                                           'Brecha_2030_optimo', 'Meses_ahorro_frontera',
                                           'Lineamiento'])

data_out.to_excel(home+"/cuadros/cuadro_lineamientos.xls", index=False)











