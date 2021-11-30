'''
Este script toma los indicadores originales (en la carpeta "raw") y los 
normaliza entre 0.1 y 0.9 utilizando sus mínimos y máximos teóricos.

'''


import matplotlib.pyplot as plt
import numpy as np
import os, copy
import pandas as pd

home =  os.getcwd()[:-4]



dfi = pd.read_excel(home+"data/raw/indicadores.xlsx", sheet_name='Indicadores')
dfm = pd.read_excel(home+"data/raw/indicadores.xlsx", sheet_name='Metadata')
df = pd.merge(left=dfi, right=dfm, left_on='Código', right_on='Código')

colYears = [year for year in df.columns if str(year).isnumeric()]
colOthers = [year for year in df.columns if not str(year).isnumeric()]

## normalize data
new_rows = []
for intex, row in df.iterrows():
    
    vals = row[colYears].values.astype(float)
    
    valsv = vals[~np.isnan(vals)]
    is_constant = np.all(valsv == valsv[0])
    
    if np.sum(~np.isnan(vals)) >= 3 and not is_constant:
    
        nvals = (row[colYears] - row['Min teórico']) / (row['Máx teórico'] - row['Min teórico'])
        
        # check that theoretical mins and max are consistent with data
        if (np.nanmax(vals) > row['Máx teórico'] or np.nanmin(vals) < row['Min teórico']):
            print( row['Código'], row['Nombre del indicador'] )
        if (row['Meta PNUD'] > row['Máx teórico'] or row['Meta PNUD'] < row['Min teórico']):
            print( row['Código'], row['Nombre del indicador'] )
        if (row['Meta MML'] > row['Máx teórico'] or row['Meta MML'] < row['Min teórico']):
            print( row['Código'], row['Nombre del indicador'] )
        
        meta_pnud = (row['Meta PNUD'] - row['Min teórico']) / (row['Máx teórico'] - row['Min teórico'])
        meta_mml = (row['Meta MML'] - row['Min teórico']) / (row['Máx teórico'] - row['Min teórico'])
                
        if row['Dirección'] == '-':
            nvals = 1-nvals
            meta_pnud = 1-meta_pnud
            meta_mml = 1-meta_mml
            
        nvals *= .8
        nvals += .1
        meta_pnud *= .8
        meta_pnud += .1
        meta_mml *= .8
        meta_mml += .1
        
        row['Meta PNUD'] = meta_pnud
        row['Meta MML'] = meta_mml
        row['Máx teórico'] = .9
        row['Min teórico'] = .1
        
        new_rows.append(nvals.tolist() + row[colOthers].values.tolist())
        
dff = pd.DataFrame(new_rows, columns=colYears+colOthers)   




# Add data on control of corruption
dfg = pd.read_excel(home+"data/raw/gov_data.xlsx", sheet_name='cc')
commonCols = [c for c in colYears if int(c) in dfg.columns.values]
cc = dfg[dfg.Code=='PER'][[int(c) for c in commonCols]].values
cc = (cc - -2.5)/(2.5 - -2.5)
cc = np.mean(cc)

dfg = pd.read_excel(home+"data/raw/gov_data.xlsx", sheet_name='rl')
commonCols = [c for c in colYears if int(c) in dfg.columns.values]
rl = dfg[dfg.Code=='PER'][[int(c) for c in commonCols]].values
rl = (rl - -2.5)/(2.5 - -2.5)
rl = np.mean(rl)

dff['control_corrupcion'] = cc
dff['estado_de_derecho'] = rl

# dfa = pd.read_excel(home+"data/raw/abrev.xlsx", sheet_name='abrev')
# df['abrev'] = dfa.abreviación

dff.to_csv(home+'data/base_norm.csv', index=False)

























