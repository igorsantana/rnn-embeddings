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

def __execute_fold(data, i, tN, metrics, queue, m2v, sm2v):
    s_embeddings, s_songs, u_sessions       = prep.split(data['train'], data['test'], m2v)
    s_s_embeddings, s_s_songs, s_u_sessions = prep.split(data['train'], data['test'], sm2v)

    m_m2vTN     = runner.execute_algo(s_embeddings, s_songs, u_sessions, 'm2vTN', tN)
    # m_sm2vTN    = runner.execute_algo(data['train'], data['test'], 'sm2vTN', tN, m2v, sm2v)
    # m_csm2vTN   = runner.execute_algo(data['train'], data['test'], 'csm2vTN', tN, 'xiami', m2v, sm2v)
    # m_csm2vUK   = runner.execute_algo(data['train'], data['test'], 'csm2vUK', tN, 'xiami', m2v, sm2v)
    queue.put(('{}_m2vTN'.format(i), m_m2vTN))
    # queue.put(('{}_sm2vTN'.format(i), m_sm2vTN))
    queue.task_done()
    return 


def execute_cv(conf):    
    topN        = int(conf['topN'])
    metrics     = conf['metrics']
    ds          = conf['dataset']
    m2v, sm2v   = __load_models()
    cv          = int(conf['cross-validation'])
    paths       = [{'train': 'tmp/cv/{}/train_{}.csv'.format(ds, i), 'test': 'tmp/cv/{}/test_{}.csv'.format(ds, i)} for i in range(cv)]
    data        = [{'train': pd.read_csv(p['train'], encoding = "utf-8"), 'test': pd.read_csv(p['test'], encoding = "utf-8")} for p in paths]
    prec    = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
    rec     = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
    fmeas   = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
    hitrate = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    q           = JoinableQueue()
    proc        = [Process(target=__execute_fold, args=(data[i], i, topN, metrics, q, m2v, sm2v)) for i in range(cv)]
    for p in proc: p.start()
    q.join()
    for i in range(cv):
        resp        = q.get()
        fold_algo   = resp[0].split('_')
        df          = resp[1]
        prec.loc[int(fold_algo[0]), fold_algo[1]]       = df['Precision'].apply(lambda v: v[0]).mean()
        rec.loc[int(fold_algo[0]), fold_algo[1]]        = df['Recall'].apply(lambda v: v[0]).mean()
        fmeas.loc[int(fold_algo[0]), fold_algo[1]]      = df['F-measure'].apply(lambda v: v[0]).mean()
        hitrate.loc[int(fold_algo[0]), fold_algo[1]]    = df['HitRate'].apply(lambda v: v[0]).mean()
    print(prec, rec, fmeas, hitrate)
    for p in proc: p.join()
    return 1