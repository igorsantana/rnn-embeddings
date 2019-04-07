import pandas as pd
import numpy as np


def split(df, cv, m2v, sm2v):
    songs       = df['song'].unique()
    sessions    = df.groupby('session').agg(lambda x: list(x))
    users       = df.groupby('user').agg(lambda x: list(x))
    s_emb       = pd.DataFrame({'music2vec': [m2v.wv[x] for x in songs], 'sessionmusic2vec': [sm2v.wv[x] for x in songs]},index=songs, columns=['music2vec','sessionmusic2vec'])
    s_songs     = pd.DataFrame({'songs': sessions.loc[:,'song']}, index=sessions.index)
    u_sess      = pd.DataFrame({'sessions': users.loc[:,'session']}, index=users.index)
    cvi         = [(dfs.index.min(), dfs.index.max()) for dfs in np.array_split(users, cv)]
    for idx, val in enumerate(cvi):
        u_sess['tt_{}'.format(idx)] = 'train'
        u_sess.loc[val[0]:val[1], 'tt_{}'.format(idx)] = 'test'
    return s_emb, s_songs, u_sess