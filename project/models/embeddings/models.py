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

def data_prep(model, data, is_doc=False):
    sequences = []
    if model == 'user':
        users = np.unique(data['user'].values).tolist()
        i = 0
        for user in users:
            u_data = data[data['user'] == user]
            sentences = [getattr(row, 'song') for row in u_data.itertuples()]
            if is_doc:
                sequences.append(TaggedDocument(words=sentences, tags=[user]))
            else:
                sequences.append(sentences)
            print('Prepping the dataset for the music2vec model [{}%]'.format(round(percentage(i, len(users)),2)), end='\r', flush=True)
            i+=1
    if model == 'session':
        last_s = data.iloc[0,3]
        sentences = []
        i = 0
        for row in data.itertuples():
            a_s = getattr(row, 'session')
            if a_s != last_s:
                if is_doc:
                    sequences.append(TaggedDocument(words=sentences, tags=[last_s]))
                else:
                    sequences.append(sentences)  
                sentences = []
            sentences.append(getattr(row, 'song'))
            last_s = a_s
            print('Prepping the dataset for the sessionmusic2vec model [{}%]'.format(round(percentage(i, len(data.index)),2)), end='\r', flush=True)
            i+=1
    return sequences
            
def music2vec(data, p):
    sequence = data_prep('user', data, False)
    return Word2Vec(sequence, size=int(p['vector_dim']),
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), sg=1, hs=0, negative=int(p['negative_sample']),
                    iter=int(p['epochs']), min_count=1, compute_loss=True)

def session_music2vec(data, p):
    sequence = data_prep('session', data, False)
    return Word2Vec(sequence, size=p['vector_dim'],
                    alpha=float(p['learning_rate']), window=int(p['window_size']),
                    sample=float(p['down_sample']), sg=1, hs=0, negative=int(p['negative_sample']),
                    iter=int(p['epochs']), min_count=1, compute_loss=True)
# def doc2vec(data):
#     sequence = data_prep('user', data, True)
#     return Doc2Vec(sequence, dm=1, vector_size=vector_dim,
#                     alpha=learning_rate, window=window_size,
#                     sample=down_sample, negative=negative_sample,
#                     epochs=epochs, min_count=1, compute_loss=True)

# def session_doc2vec(data):
#     sequence = data_prep('session', data, True)
#     return Doc2Vec(sequence, dm=1, vector_size=vector_dim,
#                     alpha=learning_rate, window=window_size,
#                     sample=down_sample, negative=negative_sample,
#                     epochs=epochs, min_count=1, compute_loss=True)

def model_runner(dataset, params):
    printlog('Checking if the models are in tmp/{}/models folder'.format(dataset))
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    should_run_m2v = True
    should_run_sm2v = True
    for file in os.listdir('tmp/{}/models/'.format(dataset)):
        print(file)
        if 'music2vec.model' == file:
            printlog('There\'s already a music2vec model in the tmp/{}/models folder.'.format(dataset))
            should_run_m2v = False
        if 'sessionmusic2vec.model' == file:
            printlog('There\'s already a sessionmusic2vec model in the tmp/{}/models folder.'.format(dataset))
            should_run_sm2v = False
    if should_run_m2v == False and should_run_sm2v == False:
        printlog('No models to run, exiting the models phase.')
        return
    df = pd.read_csv('dataset/{}/session_listening_history.csv'.format(dataset), sep = ',')
    if should_run_m2v:
        music2vec(df, params).save('tmp/{}/models/music2vec.model'.format(dataset))
    if should_run_sm2v:
        session_music2vec(df, params).save('tmp/{}/models/sessionmusic2vec.model'.format(dataset))

