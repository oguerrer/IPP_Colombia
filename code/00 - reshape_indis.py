'''

Toma los datos originales y cambia el formato de la tabla.

'''

import os, warnings
import numpy as np
import pandas as pd
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



dfi = pd.read_excel(home+"data/raw/07.09.2021_UPDATE_Datos indicadores de Colombia.xlsx", 
                    sheet_name='Indicadores ODS_Colombia_Datos', na_values=['-'])
mm = dict(zip(dfi.Indicador, dfi.IDIndicador))
dfi2 = pd.read_excel(home+"data/raw/Datos indicadores de Colombia version final.xlsx", 
                   sheet_name='Indicadores ODS_Colombia_Datos', na_values=['-'])
dfi2.loc[:, 'IDIndicador'] = [mm[c] for c in dfi2.Indicador]
targets_indis = [c.split('.')[0]+'.'+c.split('.')[1] for c in dfi2.IDIndicador]
dfi2.loc[:, 'MetaODS'] = targets_indis
dfi2.loc[:, 'ODS'] = [int(c.split('.')[0]) for c in dfi2.IDIndicador]


dfi2.rename(columns = {'2015_LB':2015, 'Instumental':'Instrumental'}, inplace = True)
dfi2.loc[:, 'Abreviatura'] = [c.replace('-', '_').replace(' ', '_') for c in dfi2.Abreviatura]

dfi3 = pd.read_excel(home+"data/raw/indicadores y sus targets.xlsx")
dfi3.loc[:, 'Abreviatura'] = [c.replace('-', '_').replace(' ', '_') for c in dfi3.Abreviatura]
abrev2inst = dict(zip(dfi3.Abreviatura, dfi3.Instumental))

dfi2.loc[:,'Instrumental'] = [abrev2inst[a] for a in dfi2.Abreviatura.values]
dfi2.to_csv(home+"data/preprocessed/indicators_reshaped.csv", index=False)





































