import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale = 1, style='whitegrid', context='paper')
palette = sns.color_palette("dark")


# fig, axs = plt.subplots(ncols=2, nrows=2)

# zero	= axs[0][0]
# um 	 	= axs[0][1]
# dois 	= axs[1][0]
# tres 	= axs[1][1]

df 					= pd.read_csv('data/results_summarized_xiami_small.csv', sep=';')
df['id'] 		= df.params.apply(lambda params: params.split('--')[:2])
df['id'] 		= df.id.apply(lambda params: '_'.join(params))
df['prec']	=	df.prec.apply(lambda prec: prec.replace(',','.')).astype(float)
df['rec']		= df.rec.apply(lambda rec: rec.replace(',','.')).astype(float)
df['f1']		= df.f1.apply(lambda f1: f1.replace(',','.')).astype(float)
df 					= df.set_index('id', drop=False)


mtn     = df[df.algo == 'm2vTN'][['id','prec','rec', 'f1']]
smtn 		= df[df.algo == 'sm2vTN'][['id','prec','rec', 'f1']]
csmtn 	= df[df.algo == 'csm2vTN'][['id','prec','rec', 'f1']]
csmuk		= df[df.algo == 'csm2vUK'][['id','prec','rec', 'f1']]

print(smtn)
# mtn 	= pd.melt(mtn, id_vars='id')
# smtn 	= pd.melt(smtn, id_vars='id')
# csmtn = pd.melt(csmtn, id_vars='id')
# csmuk = pd.melt(csmuk, id_vars='id')

# # sns.catplot(x='variable', y='value', hue='id', data=mtn, kind='bar', ax=zero, legend=False)
# sns.catplot(x='variable', y='value', hue='id', data=smtn, kind='bar')
# # sns.catplot(x='variable', y='value', hue='id', data=csmtn, kind='bar', ax=dois, legend=False)
# # sns.catplot(x='variable', y='value', hue='id', data=csmuk, kind='bar', ax=tres, legend=False)
# plt.show()
# # plt.legend(loc='upper left')
