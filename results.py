import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

np.set_printoptions(suppress=True)

df          = pd.read_csv('tmp/xiami-small/results.csv', header=None)
df.columns  = ['config', 'algo', 'prec', 'rec', 'f1']

colors      = sns.color_palette("colorblind", 4)


f = open('tmp/xiami-small/id_emb.csv', 'r')
methods = f.read()
f.close()
methods = '[{}]'.format(methods.replace('"', ''))
methods = eval(methods)
methods = dict(methods)

glove = []
barWidth = 0.2
cols = ['config','window', 'dim', 'lr', 'epochs']

for key in methods.keys():
    if 'glove' in key:
        glove.append([key] + [float(k.split('=')[1]) for k in methods[key].split(';')])

confs = pd.DataFrame(glove, columns= cols)
confs.index = confs.config
confs = confs.drop('config', 1)


df = df.merge(confs, how='left', left_on='config', right_index=True)
df['emb']   = df.config.apply(lambda x: x.split('_')[0])
df['id']    = df.config.apply(lambda x: int(x.split('_')[1]))
df          = df[df.emb == 'glove']
df          = df.drop(['emb', 'id'], axis=1)

df['window'] = df.window.apply(int)
df['dim'] = df.dim.apply(int)
df['epochs'] = df.epochs.apply(int)

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
# [0.010, 0.015, 0.020, 0.025]

for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        
        data = values[i][j][['lr', 'f1', 'config']].sort_values(by='lr').values
        x = data[:,0]
        y = data[:,1]
        labels = data[:,2]
        __labels = labels
        y50  = []
        y100 = []
        y150 = []
        y200 = []

        x50 = np.arange(x.shape[0] // 4)

        for k in range(x.shape[0] // 4):
            y50.append(y[k])
            y100.append(y[k] * 2)
            y150.append(y[k] * 3)
            y200.append(y[k] * 4)

        ax.fill_between(x50, y200, color=colors[0], label='0.025', alpha=0.2)
        ax.fill_between(x50, y150, color=colors[1], label='0.02', alpha=0.2)
        ax.fill_between(x50, y100, color=colors[2], label='0.015', alpha=0.2)
        ax.fill_between(x50, y50, color=colors[3], label='0.01', alpha=0.2)

        ax.plot(x50, y200, color=colors[0],linewidth=1)
        ax.plot(x50, y150, color=colors[1],linewidth=1)
        ax.plot(x50, y100, color=colors[2],linewidth=1)
        ax.plot(x50, y50, color=colors[3],linewidth=1)
        ax.set_title(titles[i][j])
        ax.legend(loc='upper right')
        plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=10)
        

plt.setp(axes, xticks=[r for r in range(len(__labels) // 4)], xticklabels=__labels, yticks=[x / 100 for x in np.arange(0, 101, 20)])
plt.legend()
plt.tight_layout()
plt.show()



