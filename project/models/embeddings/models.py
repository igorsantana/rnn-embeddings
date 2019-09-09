import os
import sys
import logging
import pandas               as pd
import numpy                as np
from gensim.models          import Word2Vec, Doc2Vec
from gensim.models.doc2vec  import TaggedDocument
from datetime               import datetime

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

def model_runner(dataset, params):
    m2v     =  not os.path.isfile('tmp/{}/models/music2vec.model'.format(dataset)) 
    sm2v    =  not os.path.isfile('tmp/{}/models/sessionmusic2vec.model'.format(dataset)) 
    if ((not m2v) and (not sm2v)):
        printlog('No models to run')
        return
    df = pd.read_csv('dataset/{}/session_listening_history.csv'.format(dataset), sep = ',')
    if m2v:
        printlog('Started the m2v model.')
        music2vec(df, params['music2vec']).save('tmp/{}/models/music2vec.model'.format(dataset))
    if sm2v:
        printlog('Started the sm2v model.')
        session_music2vec(df, params['music2vec']).save('tmp/{}/models/sessionmusic2vec.model'.format(dataset))
    if params['is_doc'] == True:
        printlog('Started the d2v and sd2v model.')
        doc2vec(df, params['music2vec']).save('tmp/{}/models/doc2vec.model'.format(dataset))
        session_doc2vec(df, params['music2vec']).save('tmp/{}/models/sessiondoc2vec.model'.format(dataset))
