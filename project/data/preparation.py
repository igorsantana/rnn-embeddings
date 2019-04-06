import pandas as pd
import numpy as np


def split(train ,test, model):
    train['tt']         = 'train'
    test['tt']          = 'test'
    df                  = train.append(test)
    songs               = df['song'].unique()
    sessions            = df.groupby('session').agg(lambda x: list(x))
    users               = df.groupby('user').agg(lambda x: list(x))
    s_embeddings        = pd.DataFrame({'embedding': [model.wv[x] for x in songs]},index=songs, columns=['embedding'])
    s_songs             = pd.DataFrame({'songs': sessions.loc[:,'song']}, index=sessions.index)
    u_sessions          = pd.DataFrame({'sessions': users.loc[:,'session']}, index=users.index)
    u_sessions['tt']    = df.loc[:,['user', 'tt']].groupby('user').agg({'tt': min}).iloc[:,-1]
    return s_embeddings, s_songs, u_sessions