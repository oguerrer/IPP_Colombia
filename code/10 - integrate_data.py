'''

Genera un archivo que integra a los indicadores de desarrollo con los de gobernanza.

'''

import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]




df_indis = pd.read_csv(home+"data/preprocessed/indicators_corrected.csv")
colYears = [col for col in df_indis.columns if str(col).isnumeric() and int(col)<2021]
df_indis = df_indis[df_indis[colYears].isnull().sum(axis=1) == 0]
dfi = df_indis.copy()


df_gov_cc = pd.read_csv(home+"data/preprocessed/governance_cc_normalized.csv")
df_gov_rl = pd.read_csv(home+"data/preprocessed/governance_rl_normalized.csv")



dfi['Monitoreo'] = df_gov_cc.values[:,1::].mean(axis=1)[0]
dfi['EstadoDeDerecho'] = df_gov_rl.values[:,1::].mean(axis=1)[0]



# Sort columns
scolumns = sorted(dfi.columns.values)
dfif = pd.DataFrame(dfi[scolumns].values, columns=scolumns)

dfif.to_csv(home+"data/modeling/indicators_sample_final.csv", index=False)





























































































