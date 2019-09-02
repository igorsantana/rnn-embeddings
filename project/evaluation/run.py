import sys
import logging

import numpy                            as np
import pandas                           as pd
import project.recsys.algorithms        as runner
import project.data.preparation         as prep
import project.evaluation.metrics       as m
from gensim.models                      import Word2Vec
from project.evaluation.MetricsCompiler import MetricsCompiler
from multiprocessing                    import Process, JoinableQueue

def __load_models(dataset):
    return Word2Vec.load('tmp/{}/models/music2vec.model'.format(dataset)), Word2Vec.load('tmp/{}/models/sessionmusic2vec.model'.format(dataset))

def __execute_fold(users, songs, fold, topN, k, ds):
    m_m2vTN         = runner.execute_algo('m2vTN',   users, songs, fold, topN, k, ds)
    m_sm2vTN        = runner.execute_algo('sm2vTN',  users, songs, fold, topN, k, ds)
    m_csm2vTN       = runner.execute_algo('csm2vTN', users, songs, fold, topN, k, ds)
    m_csm2vUK       = runner.execute_algo('csm2vUK', users, songs, fold, topN, k, ds)
    
def execute_cv(conf):    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    topN                    = int(conf['topN'])
    k                       = int(conf['k'])
    m2v, sm2v               = __load_models(conf['dataset'])
    df                      = pd.read_csv('dataset/{}/session_listening_history.csv'.format(conf['dataset']))
    cv                      = int(conf['cross-validation'])
    users, songs            = prep.split(df, cv, m2v, sm2v)
    metrics                 = MetricsCompiler()

    for i in range(cv):
        __execute_fold(users, songs, i, topN, k, conf['dataset'])

    # proc        = [Process(target=__execute_fold, args=(s_emb, s_songs, u_sess, i, topN, 5, q)) for i in range(cv)]

    # for p in proc: 
    #     p.start()

    # num_done = 0
    # while True:
    #     if num_done > 4: break
    #     value = q.get()
    #     fold_algo   = value[0].split('_')
    #     if fold_algo[1] == 'm2vTN': num_done+=1
    #     df          = value[1]
    #     prec.loc[int(fold_algo[0]), fold_algo[1]]       = df['Precision'].mean()
    #     rec.loc[int(fold_algo[0]), fold_algo[1]]        = df['Recall'].mean()
    #     fmeas.loc[int(fold_algo[0]), fold_algo[1]]      = df['F-measure'].mean()
    #     # hitrate.loc[int(fold_algo[0]), fold_algo[1]]    = df['HitRate'].mean()
        
    # prec.loc['mean'] = prec.mean()
    # rec.loc['mean'] = rec.mean()
    # fmeas.loc['mean'] = fmeas.mean()
    # # hitrate.loc['mean'] = hitrate.mean()

    # with open("output.txt", "w") as f:
    #     print('Precision: ', file=f)
    #     print(prec.to_string(col_space=10), end='\n\n', file=f)
    #     print('Recall: ', file=f)
    #     print(rec.to_string(col_space=10), end='\n\n', file=f)
    #     print('F-measure: ', file=f)
    #     print(fmeas.to_string(col_space=10), end='\n\n', file=f)
    #     # print('HitRate: ', file=f)
    #     # print(hitrate.to_string(col_space=10), file=f)
