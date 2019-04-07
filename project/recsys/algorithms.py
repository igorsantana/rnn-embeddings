import sys
import time
import logging
import multiprocessing                      as mp
import numpy                                as np
import pandas                               as pd
import project.recsys.Music2VecTopN         as m2vtn
import project.recsys.SessionMusic2VecTopN  as sm2vtn
import project.evaluation.metrics           as m
from scipy.spatial.distance     import cosine
from datetime                   import datetime
from operator                   import add, truediv

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
    emb     = lambda x : song_emb.loc[x]['music2vec'].tolist()
    tolist  = lambda x: list(x)
    m_values = []
    for user in users:
        sessions        = user_sess.loc[user,'sessions']    
        songs           = ses_song.loc[pd.unique(sessions),:]
        a_songs         = np.array(songs['songs'].sum())
        train_songs     = a_songs[:len(a_songs)//2]
        test_songs      = a_songs[len(a_songs)//2:]
        pref            = np.mean(emb(train_songs), axis=0)
        song_emb['cos'] = song_emb['music2vec'].apply(lambda x: 1 - cosine(x, pref))
        topn            = np.array(song_emb.nlargest(topN, 'cos').index)

        hit = hitrate(topn, test_songs)
        p   = prec(topn, test_songs)
        r   = rec(topn, test_songs)
        fm  = f_measure(p, r)

        m_values.append([hit, p, r, fm])
    printlog('Finished to evaluate users for the m2vTN algorithm')
    return pd.DataFrame(m_values, columns=['Precision', 'Recall', 'HitRate', 'F-measure'])

def run_sm2vTN(train, test, topN, sm2v):
    printlog('Started to evaluate users for the sm2vTN algorithm')
    u_s         = test.groupby('user').apply(lambda x:  x['session'].unique())
    u_s         = u_s.reset_index()
    u_s.columns = ['user','session']
    algo        = sm2vtn.SessionMusic2VecTopN(train, sm2v, topN)
    m_values    = []
    for t in u_s.itertuples():
        songs   = test[test['user'] == t[0]]['song']
        pref    = np.mean(test[test['session'] == t[1]]['song'].apply(lambda x: sm2v.wv[x]).values)
        recs    = algo.top_n(pref)
        m_values.append(m.Metrics(recs, songs.tolist()))
    printlog('Finished to evaluate users for the sm2vTN algorithm')
    return pd.DataFrame(m_values, columns=['Precision', 'Recall', 'HitRate', 'F-measure'])
            
def execute_algo(s_embeddings, s_songs, u_sessions, name, topN, i):
    if name == 'm2vTN':
        return run_m2vTN(s_embeddings, s_songs, u_sessions, topN, i)
    # if name == 'sm2vTN':
    #     return run_sm2vTN(s_embeddings, s_songs, u_sessions, topN, i)
    return 1