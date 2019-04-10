import sys
import logging
import numpy                            as np
import pandas                           as pd
import project.recsys.algorithms        as runner
import project.data.preparation   as prep
from multiprocessing                    import Process, JoinableQueue
from gensim.models                      import Word2Vec

def __load_models():
    return Word2Vec.load('tmp/models/music2vec.model'), Word2Vec.load('tmp/models/sessionmusic2vec.model')

def __execute_fold(s_emb, s_songs, u_sess, i, tN, k):
    # m_m2vTN     = runner.execute_algo(s_emb, s_songs, u_sess, 'm2vTN', tN, i)
    # m_sm2vTN    = runner.execute_algo(s_emb, s_songs, u_sess, 'sm2vTN', tN, i)
    # print(m_m2vTN)
    # print(m_sm2vTN)
    # print(m_csm2vTN)
    # m_csm2vTN   = runner.execute_algo(s_emb, s_songs, u_sess, 'csm2vTN', tN, i)
    m_csm2vUK   = runner.execute_algo(s_emb, s_songs, u_sess, 'csm2vUK', tN, i, k)
    print(m_csm2vUK)
    # queue.put(('{}_m2vTN'.format(i), m_m2vTN))
    # # queue.put(('{}_sm2vTN'.format(i), m_sm2vTN))
    # queue.task_done()
    # return 


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
    # q           = JoinableQueue()
    # proc        = [Process(target=__execute_fold, args=(s_emb, s_songs, u_sess, i, topN, q)) for i in range(cv)]
    __execute_fold(s_emb, s_songs, u_sess, 0, topN, 5)


    # for p in proc: p.start()
    # q.join()
    # for i in range(cv):
    #     resp        = q.get()
    #     fold_algo   = resp[0].split('_')
    #     df          = resp[1]
    #     prec.loc[int(fold_algo[0]), fold_algo[1]]       = df['Precision'].apply(lambda v: v[0]).mean()
    #     rec.loc[int(fold_algo[0]), fold_algo[1]]        = df['Recall'].apply(lambda v: v[0]).mean()
    #     fmeas.loc[int(fold_algo[0]), fold_algo[1]]      = df['F-measure'].apply(lambda v: v[0]).mean()
    #     hitrate.loc[int(fold_algo[0]), fold_algo[1]]    = df['HitRate'].apply(lambda v: v[0]).mean()
    # print(prec, rec, fmeas, hitrate)
    # for p in proc: p.join()
    # return 1