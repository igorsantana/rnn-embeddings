import os
import sys
import logging
import pandas               as pd
import numpy                as np
from gensim.models          import Word2Vec, Doc2Vec
from gensim.models.doc2vec  import TaggedDocument
from datetime               import datetime
from glove                  import Glove, Corpus
format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x))

def percentage(part, whole):
  return 100 * float(part)/float(whole)

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
    printlog('Prepping data')
    sentences = data_prep('user', data)
    return Word2Vec(sentences, size=int(p['vector_dim']),
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), sg=1, hs=0, negative=int(p['negative_sample']),
                    iter=int(p['epochs']), min_count=1, compute_loss=True)

def session_music2vec(data, p):
    printlog('Prepping data')
    sequence = data_prep('session', data)
    print(sequence)
    return Word2Vec(sequence, size=p['vector_dim'],
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), sg=1, hs=0, negative=int(p['negative_sample']),
                    iter=int(p['epochs']), min_count=1, compute_loss=True)
def doc2vec(data, p):
    printlog('Prepping data')
    sequence = data_prep('user_doc', data)
    return Doc2Vec(sequence, dm=1, vector_size=p['vector_dim'],
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), negative=int(p['negative_sample']),
                    epochs=int(p['epochs']), min_count=1, compute_loss=True)

def session_doc2vec(data, p):
    printlog('Prepping data')
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

def model_runner(dataset, params, embeddings):
    m2v     =  os.path.isfile('tmp/{}/models/{}.model'.format(dataset, embeddings['music2vec']['path'])) 
    sm2v    =  os.path.isfile('tmp/{}/models/{}.model'.format(dataset, 's' + embeddings['music2vec']['path'])) 
    gm2v    =  os.path.isfile('tmp/{}/models/{}.model'.format(dataset, embeddings['glove']['path'])) 
    gsm2v   =  os.path.isfile('tmp/{}/models/{}.model'.format(dataset, 's' + embeddings['glove']['path'])) 
    df = pd.read_csv('dataset/{}/session_listening_history_reduzido.csv'.format(dataset), sep = ',')

    if embeddings['glove']['usage']:
        if (not gm2v) or (not gsm2v):
            glove_user(df, params['music2vec']).save('tmp/{}/models/{}.model'.format(dataset, embeddings['glove']['path']))
            glove_session(df, params['music2vec']).save('tmp/{}/models/{}.model'.format(dataset, 's' + embeddings['glove']['path']))
        return
    if embeddings['music2vec']['usage']:
        printlog('Started the m2v model.')
        if (not m2v) or (not sm2v):
            music2vec(df, params['music2vec']).save('tmp/{}/models/{}.model'.format(dataset, embeddings['music2vec']['path']))
            session_music2vec(df, params['music2vec']).save('tmp/{}/models/{}.model'.format(dataset, 's' + embeddings['music2vec']['path']))
            return
