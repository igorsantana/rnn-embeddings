import sys
import time
import logging
import itertools
import statistics 
import multiprocessing                      as mp
import numpy                                as np
import pandas                               as pd
import project.recsys.Music2VecTopN         as m2vtn
import project.recsys.SessionMusic2VecTopN  as sm2vtn
import project.evaluation.metrics           as m
from scipy.spatial.distance                 import cdist
from datetime                               import datetime
from operator                               import add, truediv

from numpy.linalg import norm
# import SessionMusic2VecTopN             as sm2vtn
# import ContextSessionMusic2VecTopN      as csm2vtn
# import ContextSessionMusic2VecUserKNN   as csm2vuk

format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x))
st_time     = lambda _: time.time()
f_time      = lambda st: time.time() - st
hitrate     = lambda topn, test: round(sum(np.isin(topn, test)) / len(topn), 5)
prec        = lambda topn, test: round(len(np.intersect1d(topn, test)) / len(topn), 5)
rec         = lambda topn, test: round(len(np.intersect1d(topn, test)) / len(test), 5)
f_measure   = lambda prec, rec: 0.0 if (prec + rec) == 0 else round((2 * prec * rec) / (prec + rec), 5)



def run_m2vTN(song_emb, ses_song, user_sess, topN, i):
    printlog('Started to evaluate users for the m2vTN algorithm')
    users   = user_sess[user_sess['tt_{}'.format(i)] == 'test'].index.values
    def map_user(user):
        songs           = ses_song.loc[pd.unique(user_sess.loc[user,'sessions']),'songs'].values
        train           = [sess[:len(sess)//2] for sess in songs]
        test            = [sess[len(sess)//2:] for sess in songs]
        pref            = np.mean(song_emb.loc[sum(train, []), 'music2vec'], axis=0)
        song_emb['cos'] = [1 - x[0] for x in cdist(np.array(song_emb['music2vec'].tolist()), np.array([pref]), metric='cosine')]
        topn            = song_emb.nlargest(topN, 'cos').index.values
        test_songs      = sum(test, [])
        p   = prec(topn, test_songs)
        r   = rec(topn, test_songs)
        return {'HitRate': hitrate(topn, test_songs), 'Precision': p, 'Recall': r, 'F-measure': f_measure(p, r)}
    metrics = np.vectorize(map_user)(users)
    printlog('Finished to evaluate users for the m2vTN algorithm')
    return pd.DataFrame(metrics)

def run_sm2vTN(song_emb, ses_song, user_sess, topN, i):
    printlog('Started to evaluate users for the sm2vTN algorithm')
    users   = user_sess[user_sess['tt_{}'.format(i)] == 'test'].index.values

    def map_user(user):
        songs           = ses_song.loc[pd.unique(user_sess.loc[user,'sessions']),'songs'].values
        test            = [sess[len(sess)//2:] for sess in songs]
        v               = song_emb.loc[test[0], 'sessionmusic2vec']
        pref            = np.mean(v)
        song_emb['cos'] = [1 - x[0] for x in cdist(np.array(song_emb['sessionmusic2vec'].tolist()), np.array([pref]), metric='cosine')]
        topn            = song_emb.nlargest(topN, 'cos').index.values
        test_songs      = sum(test, [])
        p   = prec(topn, test_songs)
        r   = rec(topn, test_songs)
        return {'HitRate': hitrate(topn, test_songs), 'Precision': p, 'Recall': r, 'F-measure': f_measure(p, r)}

    metrics = np.vectorize(map_user)(users)
    printlog('Finished to evaluate users for the sm2vTN algorithm')
    return pd.DataFrame(metrics)
            
def execute_algo(s_embeddings, s_songs, u_sessions, name, topN, i):
    if name == 'm2vTN':
        return run_m2vTN(s_embeddings, s_songs, u_sessions, topN, i)
    if name == 'sm2vTN':
        return run_sm2vTN(s_embeddings, s_songs, u_sessions, topN, i)
    return 1