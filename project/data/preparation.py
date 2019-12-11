
import pandas as pd
import random
import numpy as np
import pickle
from os                                 import makedirs
from os.path                            import exists
from gensim.models                      import Word2Vec, Doc2Vec
from glove 								import Glove
from sklearn.model_selection            import KFold

def _rnn_load(path, songs):
    data = pickle.load(open(path, 'rb'))
    emb_dict = {}
    r       = random.choice(list(data.keys()))
    sample  = data[r]

    if sample.ndim > 1:
        for song in songs:
            emb_dict[song] = np.mean(data[song],axis=0)
    else: 
        for song in songs:
            emb_dict[song] = data[song]
    return emb_dict

def __w2v_load(path, songs):
    wv = Word2Vec.load(path).wv
    emb_dict = {}
    for song in songs:
        emb_dict[song] = wv[song]
    return emb_dict

def __g_load(path, songs):
    glove = Glove.load(path)
    emb_dict = {}
    for song in songs:
        emb_dict[song] = glove.word_vectors[glove.dictionary[song]]
    return emb_dict

def get_embeddings(path, songs):
    path_arr        = path.split('/')
    session_file    = '/'.join(path_arr[:-1] + ['s' + path_arr[-1]])
    user_file       = path
    
    if 'glove' in path:
        return __g_load(user_file, songs),__g_load(session_file, songs)
    if 'music2vec' in path:
        return __w2v_load(user_file, songs), __w2v_load(session_file, songs)
    if 'doc2vec' in path:
        return __w2v_load(user_file, songs), __w2v_load(session_file, songs)
    if 'rnn' in path:
        print(path)
        return _rnn_load(user_file, songs), _rnn_load(session_file, songs)
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
    kf                  = KFold(n_splits=conf['evaluation']['k'], shuffle=True)
    i       = 1
    kfold   = []
    for train, test in kf.split(unique_users):
        train_df = users[users.index.isin(unique_users[train])]
        test_df  = users[users.index.isin(unique_users[test])]
        train_df.to_pickle('tmp/{}/kfold/train_{}.pkl'.format(ds, i))
        test_df.to_pickle('tmp/{}/kfold/test_{}.pkl'.format(ds, i))
        kfold.append((train_df, test_df))
        i += 1 
    return kfold


    
    
    