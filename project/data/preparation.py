import logging
import pandas as pd
import numpy as np
from os                                 import makedirs
from os.path                            import exists
from gensim.models                      import Word2Vec, Doc2Vec
from glove 								import Glove
from sklearn.model_selection            import KFold
def _rnn_load(ds, path, songs):
    print('tmp/{}/models/{}.csv'.format(ds, path))
    df          = pd.read_csv('tmp/{}/models/{}.csv'.format(ds, path), sep=';', header=None)
    df.columns  = ['id', 'emb']
    df['emb']   = df['emb'].apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=','))
    df.index    = df['id']
    emb_dict    = {}
    for ix in df['id'].values:
        emb_dict[ix] = df.loc[ix, 'emb']
    return emb_dict

def __w2v_load(ds, path, songs):
    wv = Word2Vec.load('tmp/{}/models/{}.model'.format(ds, path)).wv
    emb_dict = {}
    for song in songs:
        emb_dict[song] = wv[song]
    return emb_dict

def __g_load(ds, path, songs):
    glove = Glove.load('tmp/{}/models/{}.model'.format(ds, path))
    emb_dict = {}
    for song in songs:
        emb_dict[song] = glove.word_vectors[glove.dictionary[song]]
    return emb_dict

def get_embeddings(method, ds, songs, conf):
    glove, music2vec, doc2vec, seq2seq = conf['glove'], conf['music2vec'], conf['doc2vec'], conf['seq2seq']
    if method == 'glove':
        return __g_load(ds, glove['path'], songs),__g_load(ds, 's' + glove['path'],songs)
    if method == 'music2vec':
        return __w2v_load(ds, music2vec['path'], songs), __w2v_load(ds, 's' + music2vec['path'], songs)
    if method == 'doc2vec':
        return __w2v_load(ds, doc2vec['path'], songs), __w2v_load(ds, 's' + doc2vec['path'], songs)
    if method == 'seq2seq':
        return _rnn_load(ds, seq2seq['path'], songs), _rnn_load(ds, 's' + seq2seq['path'], songs)
    return {},{} 

def get_embeddings_opt(method, ds, path, songs):
    if method == 'glove':
        return __g_load(ds, path, songs),__g_load(ds, 's' + path,songs)
    if method == 'm2v':
        return __w2v_load(ds, path, songs), __w2v_load(ds, 's' + path,songs)
    if method == 'd2v':
        return __w2v_load(ds, path, songs), __w2v_load(ds, 's' + path,songs)
    if method == 'seq2seq':
        return _rnn_load(ds, path, songs), _rnn_load(ds, 's' + path,songs)
    return {},{} 


def prepare_data(df, conf):
    ds                  = conf['evaluation']['dataset']
    path_kfold          = 'tmp/{}/kfold/'.format(ds)
    if exists(path_kfold):
        kfold = []
        for i in range(0, conf['evaluation']['k']):
            j = i + 1
            train = pd.read_pickle(path_kfold + 'train_{}.pkl'.format(j))
            test  = pd.read_pickle(path_kfold + 'test_{}.pkl'.format(j))
            kfold.append((train, test))
        return kfold
    makedirs('tmp/{}/kfold/'.format(ds))
    sessions            = df.groupby('session')['song'].apply(lambda x: x.tolist())
    users               = df.groupby('user').agg(list)
    users['history']    = users['session'].apply(lambda x: [sessions[session] for session in list(set(x))])
    users               = users.drop(['song', 'timestamp','session'], axis=1)
    unique_users        = df.user.unique()
    kf = KFold(n_splits=conf['evaluation']['k'], shuffle=True)
    i = 1
    kfold = []
    for train, test in kf.split(unique_users):
        train_df = users[users.index.isin(unique_users[train])]
        test_df  = users[users.index.isin(unique_users[test])]
        train_df.to_pickle('tmp/{}/kfold/train_{}.pkl'.format(ds, i))
        test_df.to_pickle('tmp/{}/kfold/test_{}.pkl'.format(ds, i))
        kfold.append((train_df, test_df))
        i += 1 
    return kfold


    
    
    