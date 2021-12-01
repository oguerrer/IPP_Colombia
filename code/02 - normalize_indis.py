'''

Normaliza los datos.

'''

import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/preprocessed/indicators_reshaped.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]
idxYears = [i for i, col in enumerate(df.columns) if str(col).isnumeric()]
idxMeta = np.where(df.columns=='Meta2030')[0][0]



new_rows = []
for index, row in df.iterrows():
    
    if row['Mínimo'] > np.min(row[colYears]):
        print(row)
    
    new_row = row.values.copy()
    
    maxval = row['Máximo']
    minval = row['Mínimo']
    
    new_row[idxYears] = (new_row[idxYears] - minval)/(maxval - minval)
    new_row[idxMeta] = (new_row[idxMeta] - minval)/(maxval - minval)
    
    if row['Invertir'] == 1:
        new_row[idxYears] = 1 - new_row[idxYears]
        new_row[idxMeta] = 1 - new_row[idxMeta]
    
    new_row[idxYears] = new_row[idxYears]*.8 + .1
    new_row[idxMeta] = new_row[idxMeta]*.8 + .1
    new_rows.append(new_row)


dff = pd.DataFrame(new_rows, columns=df.columns)
dff.to_csv(home+"data/preprocessed/indicators_normalized.csv", index=False)











































