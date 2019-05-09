import sys
import logging
import numpy                            as np
import pandas                           as pd
import project.recsys.algorithms        as runner
import project.data.preparation   as prep
import project.evaluation.metrics           as m
from multiprocessing                    import Process, JoinableQueue
from gensim.models                      import Word2Vec

def __load_models():
    return Word2Vec.load('tmp/models/music2vec.model'), Word2Vec.load('tmp/models/sessionmusic2vec.model')

def __execute_fold(s_emb, s_songs, u_sess, i, tN, k, queue, m2v, sm2v):
    m_m2vTN     = runner.execute_algo(s_emb, s_songs, u_sess, 'm2vTN', tN, i, k, m2v, sm2v)
    queue.put(('{}_m2vTN'.format(i), m_m2vTN))
    m_sm2vTN    = runner.execute_algo(s_emb, s_songs, u_sess, 'sm2vTN', tN, i, k, m2v, sm2v)
    queue.put(('{}_sm2vTN'.format(i), m_sm2vTN))
    m_csm2vTN   = runner.execute_algo(s_emb, s_songs, u_sess, 'csm2vTN', tN, i, k, m2v, sm2v)
    queue.put(('{}_csm2vTN'.format(i), m_csm2vTN))
    # m_csm2vUK   = runner.execute_algo(s_emb, s_songs, u_sess, 'csm2vUK', tN, i, k, m2v, sm2v)
    # queue.put(('{}_csm2vUK'.format(i), m_csm2vUK))
    

def execute_cv(conf):    
    topN                    = int(conf['topN'])
    m2v, sm2v               = __load_models()
    df                      = pd.read_csv('dataset/{}/session_listening_history.csv'.format(conf['dataset']))
    cv                      = int(conf['cross-validation'])
    s_emb, s_songs, u_sess  = prep.split(df, cv, m2v, sm2v)
    prec                    = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
    rec                     = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
    fmeas                   = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
    hitrate                 = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    q           = JoinableQueue()
    proc        = [Process(target=__execute_fold, args=(s_emb, s_songs, u_sess, i, topN, 5, q, m2v, sm2v)) for i in range(cv)]

    for p in proc: 
        p.start()

    num_done = 0
    while True:
        if num_done > 4: break
        value = q.get()
        fold_algo   = value[0].split('_')
        if fold_algo[1] == 'csm2vTN': num_done+=1
        df          = value[1]
        prec.loc[int(fold_algo[0]), fold_algo[1]]       = df['Precision'].mean()
        rec.loc[int(fold_algo[0]), fold_algo[1]]        = df['Recall'].mean()
        fmeas.loc[int(fold_algo[0]), fold_algo[1]]      = df['F-measure'].mean()
        # hitrate.loc[int(fold_algo[0]), fold_algo[1]]    = df['HitRate'].mean()
        
    prec.loc['mean'] = prec.mean()
    rec.loc['mean'] = rec.mean()
    fmeas.loc['mean'] = fmeas.mean()
    # hitrate.loc['mean'] = hitrate.mean()

    with open("output.txt", "w") as f:
        print('Precision: ', file=f)
        print(prec.to_string(col_space=10), end='\n\n', file=f)
        print('Recall: ', file=f)
        print(rec.to_string(col_space=10), end='\n\n', file=f)
        print('F-measure: ', file=f)
        print(fmeas.to_string(col_space=10), end='\n\n', file=f)
        # print('HitRate: ', file=f)
        # print(hitrate.to_string(col_space=10), file=f)

