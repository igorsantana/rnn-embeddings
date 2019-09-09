import pandas as pd
import numpy as np


def split(df, cv, m2v, sm2v):
    songs               = df['song'].unique()
    songs               = pd.DataFrame({ 'm2v': [m2v.wv[x] for x in songs], 'sm2v': [sm2v.wv[x] for x in songs] }, index=songs, columns=['m2v','sm2v'])
    sessions            = df.groupby('session')['song'].apply(lambda x: (x.name, x.tolist()))
    users               = df.groupby('user').agg(lambda x: list(x))
    users['history']    = users['session'].apply(lambda x: [sessions[session] for session in list(set(x))])
    users               = users.drop(['song', 'timestamp','session'], axis=1)
    cvi                 = [(dfs.index.min(), dfs.index.max()) for dfs in np.array_split(users, cv)]
    for idx, val in enumerate(cvi):
        users['tt_{}'.format(idx)] = 'train'
        users.loc[val[0]:val[1], 'tt_{}'.format(idx)] = 'test'
    return users, songs