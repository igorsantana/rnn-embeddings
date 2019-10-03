import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
sns.set()

np.set_printoptions(suppress=True)

df          = pd.read_csv('tmp/xiami-small-test-embeddings/results.csv', header=None)
df.columns  = ['config', 'algo', 'prec', 'rec', 'f1']

colors      = sns.color_palette("colorblind", 4)

methods ={"glove_0":"window=15;dim=100;lr=0.03;epochs=20", "glove_1":"window=15;dim=100;lr=0.03;epochs=30", "glove_2":"window=15;dim=100;lr=0.03;epochs=40", "glove_3":"window=15;dim=100;lr=0.035;epochs=20", "glove_4":"window=15;dim=100;lr=0.035;epochs=30", "glove_5":"window=15;dim=100;lr=0.035;epochs=40", "glove_6":"window=20;dim=100;lr=0.03;epochs=20", "glove_7":"window=20;dim=100;lr=0.03;epochs=30", "glove_8":"window=20;dim=100;lr=0.03;epochs=40", "glove_9":"window=20;dim=100;lr=0.035;epochs=20", "glove_10":"window=20;dim=100;lr=0.035;epochs=30", "glove_11":"window=20;dim=100;lr=0.035;epochs=40", "m2v_12":"window=15;dim=50;lr=0.015;epochs=10;down=1e-2;neg=10", "m2v_13":"window=15;dim=50;lr=0.01;epochs=10;down=1e-2;neg=10", "m2v_14":"window=15;dim=50;lr=0.015;epochs=10;down=1e-4;neg=10", "m2v_15":"window=15;dim=50;lr=0.01;epochs=10;down=1e-4;neg=10", "m2v_16":"window=20;dim=50;lr=0.015;epochs=10;down=1e-2;neg=10", "m2v_17":"window=20;dim=50;lr=0.01;epochs=10;down=1e-2;neg=10", "m2v_18":"window=20;dim=50;lr=0.015;epochs=10;down=1e-4;neg=10", "m2v_19":"window=20;dim=50;lr=0.01;epochs=10;down=1e-4;neg=10", "d2v_20":"window=15;dim=50;lr=0.03;epochs=10;down=1e-2;neg=10", "d2v_21":"window=15;dim=50;lr=0.035;epochs=10;down=1e-2;neg=10", "d2v_22":"window=15;dim=50;lr=0.03;epochs=10;down=1e-4;neg=10", "d2v_23":"window=15;dim=50;lr=0.035;epochs=10;down=1e-4;neg=10", "d2v_24":"window=20;dim=50;lr=0.03;epochs=10;down=1e-2;neg=10", "d2v_25":"window=20;dim=50;lr=0.035;epochs=10;down=1e-2;neg=10", "d2v_26":"window=20;dim=50;lr=0.03;epochs=10;down=1e-4;neg=10", "d2v_27":"window=20;dim=50;lr=0.035;epochs=10;down=1e-4;neg=10" }
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
# df['neg'] = df.neg.apply(int)

m2vtn   = df[df.algo == 'm2vTN']
m2vtn.index = m2vtn.config

sm2vtn  = df[df.algo == 'sm2vTN']
sm2vtn.index = sm2vtn.config

csm2vtn = df[df.algo == 'csm2vTN']
csm2vtn.index = csm2vtn.config

csm2vuk = df[df.algo == 'csm2vUK']
csm2vuk.index = csm2vuk.config



# m_index     = m2vtn[m2vtn.f1 > 0.0286400].index.values
# sm_index    = sm2vtn[sm2vtn.f1 > 0.0952820].index.values
# csm_index   = csm2vtn[csm2vtn.f1 > 0.0517940].index.values
# csmuk_index = csm2vuk[csm2vuk.f1 > 0.0333340].index.values

# all_better  = list(set(m_index) & set(sm_index) & set(csm_index) & set(csmuk_index))

# x           = df[df.config.isin(all_better)]

m_index     = m2vtn[m2vtn.f1 > 0.07349]
sm_index    = sm2vtn[sm2vtn.f1 > 0.12087]
csm_index   = csm2vtn[csm2vtn.f1 > 0.12057]
csmuk_index = csm2vuk[csm2vuk.f1 > 0.12082]

# all_better  = list(set(m_index) & set(sm_index) & set(csm_index) & set(csmuk_index))

# y           = df[df.config.isin(all_better)]

# print(m_index)
# print(sm_index)
# print(csm_index)
print(csmuk_index)









# print(df[df.config == 'glove_27'])
# print(Counter(x['epochs']))
# better_m2v = df[(df.window == 10) & (df.epochs == 10) & (df.neg == 10) & (df.down == 1e-3) & (df.lr == 0.02) & (df.dim== 100)].config.values[0]

# print(baseline_id)
# print(better_m2v)
# Pegar os 10 folds dessa configuração:
# window_better   = 10
# dim_better      = 100
# lr              = 0.02
# neg             = 10
# down            = 1e-3
# epochs          = 10

# e dessa também:
# window_base   = 5
# dim_base      = 300
# lr            = 0.025
# neg           = 20
# down          = 1e-3
# epochs        = 5


# values = [[m2vtn, sm2vtn], [csm2vtn, csm2vuk]]
# titles = [['m2vtn - F1', 'sm2vtn - F1'], ['csm2vtn - F1', 'csm2vuk - F1']]
# i = 0
# j = 0
# __labels = []

# fig, axes   = plt.subplots(2, 2)

# for i in range(2):
#     for j in range(2):
#         ax = axes[i, j]
        
#         data = values[i][j][['lr', 'f1', 'config']].sort_values(by='lr').values
#         x = data[:,0]
#         y = data[:,1]
#         labels = data[:,2]
#         __labels = labels
#         y50  = []
#         y100 = []
#         y150 = []
#         y200 = []

#         x50 = np.arange(x.shape[0] // 4)

#         for k in range(x.shape[0] // 4):
#             y50.append(y[k])
#             y100.append(y[k] * 2)
#             y150.append(y[k] * 3)
#             y200.append(y[k] * 4)

#         ax.fill_between(x50, y200, color=colors[0], label='0.025', alpha=0.2)
#         ax.fill_between(x50, y150, color=colors[1], label='0.02', alpha=0.2)
#         ax.fill_between(x50, y100, color=colors[2], label='0.015', alpha=0.2)
#         ax.fill_between(x50, y50, color=colors[3], label='0.01', alpha=0.2)

#         ax.plot(x50, y200, color=colors[0],linewidth=1)
#         ax.plot(x50, y150, color=colors[1],linewidth=1)
#         ax.plot(x50, y100, color=colors[2],linewidth=1)
#         ax.plot(x50, y50, color=colors[3],linewidth=1)
#         ax.set_title(titles[i][j])
#         ax.legend(loc='upper right')
#         plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=10)
        

# plt.setp(axes, xticks=[r for r in range(len(__labels) // 4)], xticklabels=__labels, yticks=[x / 100 for x in np.arange(0, 101, 20)])
# plt.legend()
# plt.tight_layout()
# plt.show()
