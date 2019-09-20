import sys
import logging
import pandas                       as pd
import numpy                        as np
from os                             import makedirs
from os.path                        import exists
from gensim.models                  import Word2Vec, Doc2Vec
from gensim.models.doc2vec          import TaggedDocument
from datetime                       import datetime
from glove                          import Glove, Corpus
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

            
def music2vec(data, p):
    sentences = data_prep('user', data)
    return Word2Vec(sentences, size=int(p['vector_dim']),
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), sg=1, hs=0, negative=int(p['negative_sample']),
                    iter=int(p['epochs']), min_count=1, compute_loss=True)

def session_music2vec(data, p):
    sequence = data_prep('session', data)
    return Word2Vec(sequence, size=p['vector_dim'],
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), sg=1, hs=0, negative=int(p['negative_sample']),
                    iter=int(p['epochs']), min_count=1, compute_loss=True)
def doc2vec(data, p):
    sequence = data_prep('user_doc', data)
    return Doc2Vec(sequence, dm=1, vector_size=p['vector_dim'],
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), negative=int(p['negative_sample']),
                    epochs=int(p['epochs']), min_count=1, compute_loss=True)

def session_doc2vec(data, p):
    sequence = data_prep('session_doc', data)
    return Doc2Vec(sequence, dm=1, vector_size=p['vector_dim'],
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), negative=int(p['negative_sample']),
                    epochs=int(p['epochs']), min_count=1, compute_loss=True)

def glove_user(data, p):
    sentences = data_prep('user', data)
    corpus = Corpus() 
    corpus.fit(sentences, window=int(p['window_size']))
    glove = Glove(no_components=p['vector_dim'], learning_rate=float(p['learning_rate']))
    glove.fit(corpus.matrix, epochs=int(p['epochs']), no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    return glove

def glove_session(data, p):
    sentences = data_prep('session', data)
    corpus = Corpus() 
    corpus.fit(sentences, window=int(p['window_size']))
    glove = Glove(no_components=p['vector_dim'], learning_rate=float(p['learning_rate']))
    glove.fit(corpus.matrix, epochs=int(p['epochs']), no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    return glove


def embeddings_opt(conf):
    ds          = conf['evaluation']['dataset']
    glove       = conf['embeddings']['glove']
    m2v         = conf['embeddings']['music2vec']
    d2v         = conf['embeddings']['doc2vec']
    seq2seq     = conf['embeddings']['seq2seq']
    df          = pd.read_csv('dataset/{}/session_listening_history.csv'.format(ds), sep = ',')
    logger      = logging.getLogger()
    if not exists('tmp'):
        makedirs('tmp')
    if not exists('tmp/{}'.format(ds)):
        makedirs('tmp/{}'.format(ds))
        makedirs('tmp/{}/models'.format(ds))
    ids_configurations = {}
    id = 0

    if glove['usage']:
        logging.info('Glove models will be generated at "%s"', 'tmp/{}/models/'.format(ds))
        g = conf['models']['glove']
        for window in g['window_size']:
            for dim in g['vector_dim']:
                for lr in g['learning_rate']:
                    for ep in g['epochs']:
                        __conf = {'window_size': window, 'vector_dim': dim, 'learning_rate': lr, 'epochs': ep}
                        glove_user(df, __conf).save('tmp/{}/models/glove_{}.model'.format(ds, id))
                        glove_session(df, __conf).save('tmp/{}/models/sglove_{}.model'.format(ds, id))
                        ids_configurations['glove_' + str(id)] = 'window={};dim={};lr={};epochs={}'.format(window, dim, lr, ep)
                        id+=1 

    if m2v['usage']:
        logging.info('Music2Vec models will be generated at "%s"', 'tmp/{}/models/'.format(ds))
        logger.setLevel(logging.ERROR)
        m = conf['models']['music2vec']
        for window in m['window_size']:
            for sample in m['negative_sample']:
                for down in m['down_sample']:
                    for lr in m['learning_rate']:
                        for ep in m['epochs']:
                            for dim in m['vector_dim']:
                                __conf = {'window_size': window, 'vector_dim': dim, 'learning_rate': lr, 'epochs': ep,
                                        'down_sample': down, 'negative_sample': sample}
                                music2vec(df, __conf).save('tmp/{}/models/m2v_{}.model'.format(ds, id))
                                session_music2vec(df, __conf).save('tmp/{}/models/sm2v_{}.model'.format(ds, id))
                                ids_configurations['m2v_' + str(id)] = 'window={};dim={};lr={};epochs={};down={};neg={}'.format(window, dim, lr, ep, down, sample)
                                id+=1
        logger.setLevel(logging.INFO)

    if d2v['usage']:
        logging.info('Doc2Vec models will be generated at "%s"', 'tmp/{}/models/'.format(ds))
        logger.setLevel(logging.ERROR)
        d = conf['models']['doc2vec']
        for window in d['window_size']:
            for sample in d['negative_sample']:
                for down in d['down_sample']:
                    for lr in d['learning_rate']:
                        for ep in d['epochs']:
                            for dim in d['vector_dim']:
                                __conf = {'window_size': window, 'vector_dim': dim, 'learning_rate': lr, 'epochs': ep,
                                        'down_sample': down, 'negative_sample': sample}
                                doc2vec(df, __conf).save('tmp/{}/models/d2v_{}.model'.format(ds, id))
                                session_doc2vec(df, __conf).save('tmp/{}/models/sd2v_{}.model'.format(ds, id))
                                ids_configurations['d2v_' + str(id)] = 'window={};dim={};lr={};epochs={};down={};neg={}'.format(window, dim, lr, ep, down, sample)
                                id+=1
        logger.setLevel(logging.INFO)
    if seq2seq['usage']:
        logging.info('RNN models will be generated at "%s"', 'tmp/{}/models/'.format(ds))
        s2s = conf['models']['seq2seq']
        for window in s2s['window_size']:
            for dim in s2s['vector_dim']:
                for batch in s2s['batch_size']:
                    for epoch in s2s['epochs']:
                        for model in s2s['model']:
                            __conf = {'window_size': window, 'vector_dim': dim, 'batch_size': batch,
                                    'epochs':epoch, 'model': model}
                            rnn_start(df, __conf, id)
                            ids_configurations['seq2seq_' + str(id)] = 'window={};dim={};bs={};epochs={};model={}'.format(window, dim, batch, epoch, model)
                            id+=1
    return ids_configurations

def embeddings(conf):
    ds          = conf['evaluation']['dataset']
    glove       = conf['embeddings']['glove']
    m2v         = conf['embeddings']['music2vec']
    d2v         = conf['embeddings']['doc2vec']
    seq2seq     = conf['embeddings']['seq2seq']
    methods     = [(glove['usage'], 'glove'), (m2v['usage'], 'music2vec'), (d2v['usage'], 'doc2vec'), (seq2seq['usage'], 'seq2seq')]
    methods     = [method[1] for method in methods if method[0]]
    df          = pd.read_csv('dataset/{}/session_listening_history.csv'.format(ds), sep = ',')
    logger      = logging.getLogger()
    logging.info('Following methods have its embeddings: %s', ', '.join(methods))
    if not exists('tmp'):
        makedirs('tmp')
    if not exists('tmp/{}'.format(ds)):
        makedirs('tmp/{}'.format(ds))
        makedirs('tmp/{}/models'.format(ds))

    if glove['usage'] and not exists('tmp/{}/models/{}.model'.format(ds, glove['path'])):
        logging.info('Glove model will be generated at "%s" and "%s"', 'tmp/{}/models/{}.model'.format(ds, glove['path']), 'tmp/{}/models/s{}.model'.format(ds, glove['path']))
        glove_user(df, conf['models']['glove']).save('tmp/{}/models/{}.model'.format(ds, glove['path']))
        glove_session(df, conf['models']['glove']).save('tmp/{}/models/s{}.model'.format(ds, glove['path']))

    if m2v['usage'] and not exists('tmp/{}/models/{}.model'.format(ds, m2v['path'])):
        logging.info('Music2Vec model will be generated at "%s" and "%s"', 'tmp/{}/models/{}.model'.format(ds, m2v['path']), 'tmp/{}/models/s{}.model'.format(ds, m2v['path']))
        
        logger.setLevel(logging.ERROR)
        music2vec(df, conf['models']['music2vec']).save('tmp/{}/models/{}.model'.format(ds, m2v['path']))
        session_music2vec(df, conf['models']['music2vec']).save('tmp/{}/models/s{}.model'.format(ds, m2v['path']))
        logger.setLevel(logging.INFO)

    if d2v['usage'] and not exists('tmp/{}/models/{}.model'.format(ds, d2v['path'])):
        logging.info('Doc2Vec model will be generated at "%s" and "%s"', 'tmp/{}/models/{}.model'.format(ds, d2v['path']), 'tmp/{}/models/s{}.model'.format(ds, d2v['path']))
        logger.setLevel(logging.ERROR)
        doc2vec(df, conf['models']['doc2vec']).save('tmp/{}/models/{}.model'.format(ds, d2v['path']))
        session_doc2vec(df, conf['models']['doc2vec']).save('tmp/{}/models/s{}.model'.format(ds, d2v['path']))
        logger.setLevel(logging.INFO)
    if seq2seq['usage'] and not exists('tmp/{}/models/{}.csv'.format(ds, seq2seq['path'])):
        logging.info('RNN model will be generated at "%s" and "%s"', 'tmp/{}/models/{}.csv'.format(ds, seq2seq['path']), 'tmp/{}/models/s{}.csv'.format(ds, seq2seq['path']))
        rnn_start(df, conf, None)
    
    return methods