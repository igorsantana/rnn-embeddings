import sys

import pickle
import pandas                       as pd
import numpy                        as np
from os                             import makedirs
from os.path                        import exists
from gensim.models                  import Word2Vec, Doc2Vec
from gensim.models.doc2vec          import TaggedDocument
from datetime                       import datetime
from glove                          import Glove, Corpus
from project.models.rnn             import rnn
from project.models.setups          import Setups
from project.models.seq2seq         import start as rnn_start

def data_prep(model, df):
    if model == 'user':
        return df.groupby(by='user')['song'].apply(list).values.tolist()
    if model == 'user_doc':
        return df.groupby(by='user')['song'].apply(lambda x: TaggedDocument(words=x.tolist(), tags=[x.name])).values.tolist()
    if model == 'session':
        return df.groupby(by='session')['song'].apply(list).values.tolist()
    if model == 'session_doc':
        return df.groupby(by='session')['song'].apply(lambda x: TaggedDocument(words=x.tolist(), tags=[x.name])).values.tolist()

def music2vec(data, w2v_type, dim, lr, window, down, neg_sample, epochs):
    sentences = data_prep(w2v_type, data)
    return Word2Vec(sentences, size=dim, alpha=lr, window=window, sample=down,
                    sg=1, hs=0, negative=neg_sample, iter=epochs, min_count=1, compute_loss=True)

def doc2vec(data, d2v_type, dim, lr, window, down, neg_sample, epochs):
    sequence = data_prep(d2v_type, data)
    return Doc2Vec(sequence, dm=1, vector_size=dim, alpha=lr, window=window, sample=down,
                    negative=neg_sample, epochs=epochs, min_count=1, compute_loss=True)

def glove(data, glove_type, window, dim, lr, epochs):
    sentences = data_prep(glove_type, data)
    corpus = Corpus() 
    corpus.fit(sentences, window=window)
    glove = Glove(no_components=dim, learning_rate=lr)
    glove.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    return glove

def embeddings(df, conf):
    ds          = conf['evaluation']['dataset']
    cwd         = 'tmp/{}/models'.format(ds)

    if not exists(cwd):
        makedirs(cwd)
        
    setups      = Setups(conf)
    generators  = setups.get_generators()

    c_id = 0
    setups_id = []
    for method, generator in generators:
        if method == 'rnn':
            for s in generator:
                to_str  = setups.setup_to_string(c_id, s, method)

                path    = '{}/{}__{}.pickle'.format(cwd, method, c_id)
                path_s  = '{}/s{}__{}.pickle'.format(cwd, method, c_id)

                if not exists(path):
                    user, session = rnn(df, ds, s['model'], s['window'], s['epochs'], 
                                        s['batch'], s['dim'], s['num_units'], s['bidi'])
                    fu = open(path, 'wb')
                    fs = open(path_s, 'wb')

                    pickle.dump(user, fu, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(session, fs, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    fu.close()
                    fs.close()

                setups_id.append([c_id, to_str, path])
                c_id+=1
        if method == 'music2vec':
            for s in generator:
                to_str  = setups.setup_to_string(c_id, s, method)

                path    = '{}/{}__{}.model'.format(cwd, method, c_id)
                path_s  = '{}/s{}__{}.model'.format(cwd, method, c_id) 

                if not exists(path):

                    m2v  = music2vec(df,'user', s['dim'], s['lr'], s['window'], s['down'], s['neg_sample'], s['epochs'])
                    sm2v = music2vec(df,'session', s['dim'], s['lr'], s['window'], s['down'], s['neg_sample'], s['epochs'])

                    m2v.save(path)
                    sm2v.save(path_s)

                setups_id.append([c_id, to_str, path])

                c_id+=1
        if method == 'doc2vec':
            for s in generator:
                to_str  = setups.setup_to_string(c_id, s, method)
                path    = '{}/{}__{}.model'.format(cwd, method, c_id)
                path_s  = '{}/s{}__{}.model'.format(cwd, method, c_id) 

                if not exists(path):

                    d2v = doc2vec(df,'user_doc', s['dim'], s['lr'], s['window'], s['down'], s['neg_sample'], s['epochs'])
                    sd2v = doc2vec(df,'session_doc', s['dim'], s['lr'], s['window'], s['down'], s['neg_sample'], s['epochs'])

                    d2v.save(path)
                    sd2v.save(path_s)

                setups_id.append([c_id, to_str, path])

                c_id+=1
        if method == 'glove':
            for s in generator:
                to_str  = setups.setup_to_string(c_id, s, method)
                path    = '{}/{}__{}.model'.format(cwd, method, c_id)
                path_s  = '{}/s{}__{}.model'.format(cwd, method, c_id) 

                if not exists(path):

                    glv = glove(df, 'user', s['window'], s['dim'], s['lr'], s['epochs'])
                    sglv = glove(df, 'session', s['window'], s['dim'], s['lr'], s['epochs'])
                    
                    glv.save(path)
                    sglv.save(path_s)

                setups_id.append([c_id, to_str, path])

                c_id+=1

    setups_id = np.stack(setups_id, axis=0)
    
    np.save('{}/ids'.format(cwd), setups_id)
    