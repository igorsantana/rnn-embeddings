import pandas as pd
import numpy as np
from gensim.models                      import Word2Vec, Doc2Vec
from glove 								import Glove

def _rnn_load(ds, path, songs):
    df          = pd.read_csv('tmp/{}/models/{}'.format(ds, path), sep=';', header=None)
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

def get_embeddings(conf, ds, songs):
    glove, music2vec, doc2vec, rnn = conf['glove'], conf['music2vec'], conf['doc2vec'], conf['rnn']
    if glove['usage']:
        return __g_load(ds, glove['path'],songs),__g_load(ds, 's' + glove['path'],songs)
    if music2vec['usage']:
        return __w2v_load(ds, music2vec['path'], songs), __w2v_load(ds, 's' + music2vec['path'], songs)
    if doc2vec['usage']:
        return __w2v_load(ds, doc2vec['path'], songs), __w2v_load(ds, 's' + doc2vec['path'], songs)
    if rnn['usage']:
        return _rnn_load(ds, rnn['path'], songs), _rnn_load(ds, rnn['session_path'], songs)
    return {},{} 

def split(df, cv, embeddings_conf, dataset):
    songs               = df['song'].unique()
    m2v, sm2v           = get_embeddings(embeddings_conf, dataset, songs)
    songs               = pd.DataFrame({ 'm2v': [m2v[x] for x in songs], 'sm2v': [sm2v[x] for x in songs]},index=songs, columns=['m2v','sm2v'])
    sessions            = df.groupby('session')['song'].apply(lambda x: (x.name, x.tolist()))
    users               = df.groupby('user').agg(lambda x: list(x))
    users['history']    = users['session'].apply(lambda x: [sessions[session] for session in list(set(x))])
    users               = users.drop(['song', 'timestamp','session'], axis=1)
    cvi                 = [(dfs.index.min(), dfs.index.max()) for dfs in np.array_split(users, cv)]
    for idx, val in enumerate(cvi):
        users['tt_{}'.format(idx)] = 'train'
        users.loc[val[0]:val[1], 'tt_{}'.format(idx)] = 'test'
    return users, songs


    
    
    