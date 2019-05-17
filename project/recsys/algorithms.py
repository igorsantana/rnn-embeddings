import sys
import time
import math
import logging
import itertools
import operator
import statistics                           as s
import multiprocessing                      as mp
import numpy                                as np
import pandas                               as pd
import project.evaluation.metrics           as m
from scipy.spatial.distance                 import cdist, cosine, squareform, pdist
from datetime                               import datetime
from operator                               import add, truediv
from sklearn.metrics.pairwise               import cosine_similarity
from numpy.linalg                           import norm

format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x))
st_time     = lambda _: time.time()
f_time      = lambda st: time.time() - st



def run_m2vTN(song_emb, ses_song, user_sess, topN, i, mv):
    printlog('Started to evaluate users for the m2vTN algorithm')
    users         = user_sess[user_sess['tt_{}'.format(i)] == 'test'].index.values
    metrics_final = []
    def map_user(user):
        sessions = ses_song.loc[user_sess.loc[user, 'sessions'], 'songs'].values
        train    = [session[:len(session)//2] for session in sessions]
        pref     = np.mean(song_emb.loc[sum(train, []), 'music2vec'], axis=0)
        metrics  = []
        for session in sessions:
            if len(session) > 1:
                test    = session[len(session)//2:]
                topn    = mv.similar_by_vector(pref, topn=topN, restrict_vocab=None)
                topn    = [x[0] for x in topn]
                p       = m.Precision(topn, test)
                r       = m.Recall(topn, test)
                fm      = m.FMeasure(p, r)
                metrics.append([p, r, fm])
        mean = np.mean(metrics, axis=0)
        return {'Precision': mean[0], 'Recall': mean[1], 'F-measure': mean[2]}
    metrics_final = np.vectorize(map_user)(users)
    printlog('Finished to evaluate users for the m2vTN algorithm')
    return pd.DataFrame(list(metrics_final))

def run_sm2vTN(song_emb, ses_song, user_sess, topN, i, mv):
    printlog('Started to evaluate users for the sm2vTN algorithm')
    users         = user_sess[user_sess['tt_{}'.format(i)] == 'test'].index.values
    metrics_final = []
    def map_user(user):
        sessions = ses_song.loc[user_sess.loc[user, 'sessions'], 'songs'].values
        metrics  = []
        for session in sessions:
            if len(session) > 1:
                test            = session[len(session)//2:]
                pref            = np.mean(song_emb.loc[session[:len(session)//2], 'sessionmusic2vec'], axis=0)
                topn            = mv.similar_by_vector(pref, topn=topN, restrict_vocab=None)
                topn            = [x[0] for x in topn]
                p               = m.Precision(topn, test)
                r               = m.Recall(topn, test)
                fm              = m.FMeasure(p, r)
                metrics.append([p, r, fm])
        mean = np.mean(metrics, axis=0)
        return {'Precision': mean[0], 'Recall': mean[1], 'F-measure': mean[2]}

    metrics_final = np.vectorize(map_user)(users)
    printlog('Finished to evaluate users for the sm2vTN algorithm')
    return pd.DataFrame(list(metrics_final))
    

def run_csm2vTN(song_emb, ses_song, user_sess, topN, i, mv, smv):
    printlog('Started to evaluate users for the csm2vTN algorithm')
    users         = user_sess[user_sess['tt_{}'.format(i)] == 'test'].index.values
    metrics_final = []
    
    def map_user(user):
        sessions    = ses_song.loc[user_sess.loc[user, 'sessions'], 'songs'].values
        all_songs   = song_emb.index.values.tolist()
        train       = [session[:len(session)//2] for session in sessions]
        u_pref      = np.mean(song_emb.loc[sum(train, []), 'music2vec'], axis=0)
        u_sim       = [1 - dist for dist in mv.wv.distances(u_pref, all_songs)]
        metrics     = []
        for session in sessions:
            if len(session) > 1:
                test            = session[len(session)//2:]
                c_pref          = np.mean(song_emb.loc[session[:len(session)//2], 'sessionmusic2vec'], axis=0)
                c_sim           = [1 - dist for dist in mv.wv.distances(c_pref, all_songs)]
                sim             = list(zip(u_sim, c_sim))
                sim             = [x[0] + x[1] for x in sim]
                songs_sim       = list(zip(all_songs, sim))
                songs_sim.sort(key= operator.itemgetter(1), reverse=True)
                topn            = [x[0] for x in songs_sim[:topN]]
                p               = m.Precision(topn, test)
                r               = m.Recall(topn, test)
                fm              = m.FMeasure(p, r)
                metrics.append([p, r, fm])
        mean = np.mean(metrics, axis=0)
        return {'Precision': mean[0], 'Recall': mean[1], 'F-measure': mean[2]}

    metrics_final = np.vectorize(map_user)(users)
    printlog('Finished to evaluate users for the csm2vTN algorithm')
    print(metrics_final)
    return pd.DataFrame(list(metrics_final))

def run_csm2vUK(song_emb, ses_song, user_sess, topN, i, k):
    printlog('Started to evaluate users for the csm2vUK algorithm')
    users   = user_sess[user_sess['tt_{}'.format(i)] == 'test'].index.values

    def u_sim(u, v):
        d = math.sqrt(int(u[-1] - 10000) * int(v[-1]) - 10000) + (1 - cosine(u[:len(u)-1], v[:len(v)-1]))
        return 0 if not d else (1 / d)

    def user_pref_fn(u):
        songs           = ses_song.loc[pd.unique(user_sess.loc[u,'sessions']),'songs'].values
        train           = [sess[:len(sess)//2] for sess in songs]
        pref            = np.mean(song_emb.loc[sum(train, []), 'music2vec'], axis=0)
        return pref
        # u_songs     = ses_song.loc[pd.unique(user_sess.loc[u,'sessions']),'songs'].values.tolist()
        # return np.mean(song_emb.loc[sum([sess[:len(sess)//2] for sess in u_songs], []), 'sessionmusic2vec'])

    user_pref   = pd.DataFrame([np.append(user_pref_fn(user), [10000 + len(ses_song.loc[pd.unique(user_sess.loc[user,'sessions']),'songs'].values.tolist())]) for user in user_sess.index.values], index=user_sess.index, columns=range(0, 301))
    print(user_pref)
    sim_matrix  = pd.DataFrame(squareform(pdist(user_pref, u_sim)), index=user_sess.index, columns=user_sess.index)
    print(sim_matrix)
    def map_user(user):
        pref            = user_pref_fn(user)
        songs           = ses_song.loc[pd.unique(user_sess.loc[user,'sessions']),'songs'].values
        test            = [sess[len(sess)//2:] for sess in songs]
        k_sim           = sim_matrix.nlargest(k, user).index.values
        s_k_sim         = [sum(ses_song.loc[pd.unique(user_sess.loc[user,'sessions']),'songs'].values, []) for user in k_sim]
        users_listened  = lambda m: [k_sim[i] for i, song_arr in enumerate(s_k_sim) if m in song_arr]
        
        song_emb['cos']     = [1 - x[0] for x in cdist(np.array(song_emb['sessionmusic2vec'].tolist()), np.array([pref]), metric='cosine')]
        song_emb['u_sim']   = [[sim_matrix.loc[user, y] for y in users_listened(x)] for x in song_emb.index.values]
        pref_values = []
        for song in song_emb.index.values:
            u_sims      = song_emb.loc[song, 'u_sim']
            cos         = song_emb.loc[song, 'cos']
            len_sim     = len(u_sims)
            pref_values.append(sum([sim / (float(len_sim) + cos) for sim in u_sims]))
        song_emb['pref']    = pref_values
        topn                = list(song_emb.nlargest(topN, 'pref').index.values)
        test_songs      = sum(test, [])
        p   = prec(topn, test_songs)
        r   = rec(topn, test_songs)
        return {'Precision': p, 'Recall': r, 'F-measure': f_measure(p, r)}

    metrics = np.vectorize(map_user)(users)
    printlog('Finished to evaluate users for the csm2vUK algorithm')
    return pd.DataFrame(list(metrics))


def execute_algo(s_embeddings, s_songs, u_sessions, name, topN, i, k, m2v, sm2v):
    if name == 'm2vTN':
        return run_m2vTN(s_embeddings, s_songs, u_sessions, topN, i, m2v)
    if name == 'sm2vTN':
        return run_sm2vTN(s_embeddings, s_songs, u_sessions, topN, i, sm2v)
    if name == 'csm2vTN':
        return run_csm2vTN(s_embeddings, s_songs, u_sessions, topN, i, m2v, sm2v)
    if name == 'csm2vUK':        
        return run_csm2vUK(s_embeddings, s_songs, u_sessions, topN, i, k)
    return 0
    
    