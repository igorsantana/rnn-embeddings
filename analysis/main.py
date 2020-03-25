import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale =1, style='whitegrid', context='paper')
colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", '#f1c40f']
palette = sns.color_palette(colors)

df 				= pd.read_csv('data/xiami.csv', sep='\t')
df['id'] 		= df.params

mtn         = df[df.algo == 'm2vTN'][['id','prec','rec', 'f1']]
mtn 		= pd.DataFrame(mtn.groupby(by='id').mean())
mtn['id'] 	= mtn.index

smtn 		= df[df.algo == 'sm2vTN'][['id','prec','rec', 'f1']]
smtn 		= pd.DataFrame(smtn.groupby(by='id').mean())
smtn['id'] 	= smtn.index

csmtn 	    = df[df.algo == 'csm2vTN'][['id','prec','rec', 'f1']]
csmtn 		= pd.DataFrame(csmtn.groupby(by='id').mean())
csmtn['id']	= csmtn.index

csmuk		= df[df.algo == 'csm2vUK'][['id','prec','rec', 'f1']]
csmuk 		= pd.DataFrame(csmuk.groupby(by='id').mean())
csmuk['id']	= csmuk.index

mtn.sort_index(ascending=False, inplace=True)
smtn.sort_index(ascending=False, inplace=True)
csmtn.sort_index(ascending=False, inplace=True)
csmuk.sort_index(ascending=False, inplace=True)

melt_mtn 	= pd.melt(mtn, id_vars='id')
melt_smtn 	= pd.melt(smtn, id_vars='id')
melt_csmtn 	= pd.melt(csmtn, id_vars='id')
melt_csmuk 	= pd.melt(csmuk, id_vars='id')

fig, axes = plt.subplots(2, 2, figsize=(25, 25))

a1 = sns.catplot(x='variable', y='value', hue='id', data=melt_mtn, kind='bar', palette=palette, ax=axes[0][0])
a2 = sns.catplot(x='variable', y='value', hue='id', data=melt_smtn, kind='bar', palette=palette, ax=axes[0][1])
a3 = sns.catplot(x='variable', y='value', hue='id', data=melt_csmtn, kind='bar', palette=palette, ax=axes[1][0])
a4 = sns.catplot(x='variable', y='value', hue='id', data=melt_csmuk, kind='bar', palette=palette, ax=axes[1][1])

plt.close(2)
plt.close(3)
plt.close(4)
plt.close(5)

titles = ['M-TN', 'SM-TN', 'CSM-TN', 'CSM-UK']

last = axes.flatten()[-1]
handles, labels = last.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')

i=0
for ax in axes.flatten():
	ax.get_legend().remove()
	ax.set(yticks=np.arange(0, 0.21, 0.025))
	ax.set(xlabel='Métrica Utilizada', ylabel='Valor')
	ax.set(title=titles[i])
	i+=1
	

plt.subplots_adjust(hspace=0.4)
fig.suptitle('SPLIT 30; SESSIONS >= {}; 100% Dataset (n=full)'.format(10), fontsize=18, y=.98)
plt.show()


