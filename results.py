import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

np.set_printoptions(suppress=True)

df          = pd.read_csv('tmp/xiami-small/results.csv', header=None)
df.columns  = ['config', 'algo', 'prec', 'rec', 'f1']

colors      = sns.color_palette("colorblind", 100)


f = open('tmp/xiami-small/id_emb.csv', 'r')
methods = f.read()
f.close()
methods = '[{}]'.format(methods.replace('"', ''))
methods = eval(methods)
methods = dict(methods)

m2v = []
barWidth = 0.015
cols = ['config','window', 'dim', 'lr', 'epochs', 'down', 'neg']

for key in methods.keys():
    if 'm2v' in key:
        m2v.append([key] + [float(k.split('=')[1]) for k in methods[key].split(';')])


confs = pd.DataFrame(m2v, columns= cols)
confs.index = confs.config
confs = confs.drop('config', 1)


df = df.merge(confs, how='left', left_on='config', right_index=True)
df['emb']   = df.config.apply(lambda x: x.split('_')[0])
df['id']    = df.config.apply(lambda x: int(x.split('_')[1]))
df          = df[df.emb == 'm2v']
df          = df.drop(['emb', 'id'], axis=1)
df['window'] = df.window.apply(int)
df['dim'] = df.dim.apply(int)
df['epochs'] = df.epochs.apply(int)
df['neg'] = df.neg.apply(int)

m2vtn   = df[df.algo == 'm2vTN']
sm2vtn  = df[df.algo == 'sm2vTN']
csm2vtn = df[df.algo == 'csm2vTN']
csm2vuk = df[df.algo == 'csm2vUK']


values = [[m2vtn, sm2vtn], [csm2vtn, csm2vuk]]
titles = [['m2vtn - F1', 'sm2vtn - F1'], ['csm2vtn - F1', 'csm2vuk - F1']]
i = 0
j = 0
__labels = []

fig, axes   = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        
        data = values[i][j][['neg', 'f1', 'config']].sort_values(by='neg').values
        x = data[:,0]
        n_values = np.unique(x)
        y = data[:,1]
        
        labels = data[:,2]
        __labels = labels


        ind     = np.arange(n_values.shape[0])
        ind2    = np.arange(n_values.shape[0])

        ax.xaxis.set_ticks(ind)
        ax.xaxis.set_ticklabels(n_values)

        ax.bar(ind, [y[0], y[0] * 2, y[0] * 3, y[0] * 4, y[0] * 5], width=barWidth, color=colors[7])
        fg =  (x.shape[0] // n_values.shape[0]) // 2
        c = 0
        for k in range(1, fg):
            ax.bar(ind2 - barWidth, [y[k], y[k] * 2, y[k] * 3, y[k] * 4, y[k] * 5], width=barWidth, color=colors[c])
            ind2 = np.array([x - barWidth for x in ind2])
            c+=1

        for k in range(fg, fg * 2):
            ax.bar(ind + barWidth, [y[k], y[k] * 2, y[k] * 3, y[k] * 4, y[k] * 5], width=barWidth, color=colors[c])
            ind = np.array([x + barWidth for x in ind])
            c+=1

        ax.set_title(titles[i][j])
        ax.legend(loc='upper right')
        plt.setp(ax.get_xticklabels(), fontsize=10)
        


plt.setp(axes, yticks=[x / 100 for x in np.arange(0, 101, 5)])
# plt.legend()
plt.tight_layout()
plt.suptitle('Resultados relativos ao número de dimensões utilizada para se obter os resultados')
plt.show()



