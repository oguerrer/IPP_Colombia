'''

Grafica los resultados de optimizar el presupuesto entre los indicadores sensibles.

'''


import matplotlib.pyplot as plt
import numpy as np
import os, copy, re, csv
import pandas as pd


home =  os.getcwd()[:-4]

path = home+'/figuras/'


df = pd.read_csv(home+"data/modeling/indicators_sample_final.csv")
colYears = np.array([c for c in df.columns if c.isnumeric()])

file = open(home+"/data/misc/sdg_colors.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()




dfi = pd.read_csv(home+"data/sims/prospective.csv")
all_vals = dfi.values[:,1::]




# Parameters
df_params = pd.read_csv(home+"data/modeling/parameters.csv")
alphas = df_params.alpha.values.copy()
betas = df_params.beta.values.copy()
T = int(df_params['T'].values[0])
scalar = df_params.scalar.values[0]
min_value = df_params.min_value.values[0]
num_years = df_params.years.values[0]
sub_periods = int(T/num_years)
years_forward = 10


# Indicators
N = len(df)
R = df.Instrumental.values.copy()
n = R.sum()


# Budget
df_expt = pd.read_csv(home+"data/modeling/budget_target.csv")
annual_average_bud = df_expt[colYears].mean(axis=1).values
Bs_retro_tot = np.tile(annual_average_bud, (years_forward,1)).T
Bs = [[] for g in range(len(df_expt))]
for i, year in enumerate(range(2021, 2020+years_forward+1)):
    for j in range(sub_periods):
        for index, row in df_expt.iterrows():
            Bs[index].append( Bs_retro_tot[index,i]/sub_periods )
Bs0 = np.array(Bs)
usdgs = df_expt.values[:,0]
sdg2index = dict(zip(usdgs, range(len(usdgs))))
sdgs = df.MetaODS.values
B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])




dfi_pro = pd.read_csv(home+"data/sims/prospective.csv")
fin_vals_pro = dfi_pro.values[:,years_forward*sub_periods]

dfi_fro = pd.read_csv(home+"data/sims/frontier.csv")
fin_vals_fro = dfi_fro.values[:,years_forward*sub_periods]

convergence = dfi_fro.values[:,1:years_forward*sub_periods+1] >= np.tile(fin_vals_pro, (dfi_fro.values[:,1:years_forward*sub_periods+1].shape[1],1)).T
savings = np.array([12*(years_forward*sub_periods - np.where(conv)[0][0])/sub_periods for conv in convergence])
sensible = np.where( (savings >= 60) & (fin_vals_pro < df.Meta2030*scalar))[0]
targets_sensible = np.where(df_expt.MetaODS.isin(df.MetaODS.values[sensible]).values)[0]
budget_sensible = annual_average_bud[targets_sensible].sum()
indis_sensible = np.where(df.MetaODS.isin(df_expt.MetaODS.values[targets_sensible]))[0]


target_budget_dict = dict(zip(df_expt.MetaODS.values, annual_average_bud/budget_sensible))


dfx_opt = pd.read_csv(home+"data/modeling/optimal_budget_0.csv")
B_opt = dfx_opt.gasto.values
target_budget_dict_0 = dict(zip(df_expt.MetaODS.values, B_opt/budget_sensible))

dfx_opt = pd.read_csv(home+"data/modeling/optimal_budget_100.csv")
B_opt = dfx_opt.gasto.values
target_budget_dict_100 = dict(zip(df_expt.MetaODS.values, B_opt/(budget_sensible*2)))

dfx_opt = pd.read_csv(home+"data/modeling/optimal_budget_1000.csv")
B_opt = dfx_opt.gasto.values
target_budget_dict_1000 = dict(zip(df_expt.MetaODS.values, B_opt/(budget_sensible*10)))


new_rows2 = []
target = ''
for index, row in df.iterrows():
    if index in indis_sensible:
        target = row.MetaODS
        
        budget_target = '{:0.3f}'.format(100*target_budget_dict[row.MetaODS])
        budget_target_0 = '{:0.3f}'.format(100*target_budget_dict_0[row.MetaODS])
        budget_target_100 = '{:0.3f}'.format(100*target_budget_dict_100[row.MetaODS])
        budget_target_1000 = '{:0.3f}'.format(100*target_budget_dict_1000[row.MetaODS])
        n_indis = str(sum(df.MetaODS==row.MetaODS))
        
        if index in sensible:
            new_rows2.append([row.MetaODS, row.Abreviatura, 'sí',
                  budget_target, budget_target_0, budget_target_100, budget_target_1000])
        else:
            new_rows2.append([row.MetaODS, row.Abreviatura, 'no',
                  budget_target, budget_target_0, budget_target_100, budget_target_1000])
            
dfi = pd.DataFrame(new_rows2, columns=['Target', 'Indicador', 'Sensible', 'Proporción original', 'Proporción bajo óptimo', 'Proporción bajo óptimo x2', 'Proporción bajo óptimo x10'])
dfi.to_excel(home+'/cuadros/cuadro_2.xls')











metas = df.Meta2030.values*scalar

dfi_pro = pd.read_csv(home+"data/sims/prospective.csv")
fin_vals_pro = dfi_pro.values[:,sub_periods*10]
gap_pro = metas-fin_vals_pro
gap_pro[gap_pro<0] = 0


for interest in [0, 100, 1000]:

    dfi_opt = pd.read_csv(home+"data/sims/optimal_prospective_"+str(interest)+".csv")
    fin_vals_opt = dfi_opt.values[:,sub_periods*10]
    gap_opt = metas-fin_vals_opt
    gap_opt[gap_opt<0] = 0
    
    plt.figure(figsize=(6,4))
    for index, row in df.iterrows():
        if index in indis_sensible:
            if row.Instrumental==0:
                plt.plot(gap_pro[index], gap_opt[index], '.', mec='w', markersize=20,
                      mfc=colors_sdg[int(row['MetaODS'].split('.')[0])])
            else:
                plt.plot(gap_pro[index], gap_opt[index], 'p', mec='w', markersize=12,
                      mfc=colors_sdg[int(row['MetaODS'].split('.')[0])])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='y')
    plt.xlabel('brechas esperadas en 2030', fontsize=14)
    plt.ylabel('brechas bajo optimización', fontsize=14)
    plt.tight_layout()
    plt.savefig(path+'brechas_opt_'+str(interest)+'.pdf')
    plt.show()













dft = pd.read_csv(home+"data/sims/optimal_prospective_0.csv")
all_vals_t = dft.values[:,1::]

on_time=[]
late=[]
unfeasible=[]
feasible=[]
i=0
for index, row in df.iterrows():
    meta = scalar*row.Meta2030
    gaps = meta - all_vals_t[i]
    reaches = np.where(gaps<=0)[0]
    if len(reaches) > 0 and reaches[0]/sub_periods <= 10:
        on_time.append(index)
        feasible.append(index)
    elif len(reaches) > 0 and reaches[0]/sub_periods <= 20:
        late.append(index)
        feasible.append(index)
    else:
        unfeasible.append(index)
    i+=1

fig = plt.figure(figsize=(4.5,4.5))
ax = fig.add_subplot(111)
ax.axis('equal')
width = 0.3

cm = plt.get_cmap("tab20c")
cout = cm(np.arange(3)*4)
pie, texts, pcts = ax.pie([len(on_time), len(late), len(unfeasible)], radius=1-width, startangle=90, counterclock=False,
                          colors=['lightgrey', 'grey', 'black'], autopct='%.0f%%', pctdistance=0.79)
plt.setp( pie, width=width, edgecolor='white')
plt.setp(pcts[0], color='black')
plt.setp(pcts[1], color='black')
plt.setp(pcts[2], color='white')
ax.legend(pie, ['menos de\n10 años', '10 a 20 años', 'más de\n20 años'],
          loc="center",
          bbox_to_anchor=(.25, .5, 0.5, .0),
          fontsize=8,
          frameon=False
          )

cin = [colors_sdg[df.loc[c].ODS] for c in on_time] + [colors_sdg[df.loc[c].ODS] for c in late] + [colors_sdg[df.loc[c].ODS] for c in unfeasible]
labels = [df.loc[c].Abreviatura for c in on_time] + [df.loc[c].Abreviatura for c in late] + [df.loc[c].Abreviatura for c in unfeasible]
pie2, _ = ax.pie(np.ones(len(df)), radius=1, colors=cin, labels=labels, rotatelabels=True, shadow=False, counterclock=False,
                  startangle=90, textprops=dict(va="center", ha='center', rotation_mode='anchor', fontsize=5), 
                  labeldistance=1.17)
plt.setp( pie2, width=width, edgecolor='none')
plt.tight_layout()
plt.savefig(path+'dona_convergencia_optimo.pdf')
plt.show()







































