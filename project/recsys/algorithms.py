import os
import sys
import time
import yaml
import multiprocessing                      as mp
import numpy                                as np
from project.evaluation.metrics             import get_metrics
from datetime                               import datetime
from sklearn.metrics.pairwise               import cosine_similarity

def execute_algo(train, test, songs, topN, k_sim, data):

    m2vTN   = []
    sm2vTN  = []
    csm2vTN = [] 
    csm2vUK = []
    
    u_songs  = data.us_matrix()
    users    = data.uu_matrix()

    def pref(u, k_similar, song):
        listened_to = [(k, u_songs[k, data.song_ix(song)] == 1) for k in k_similar]
        sum_sims = 0
        for u_k, listen in listened_to:
            if listen == True:
                sum_sims += users[u][u_k] / [v[1] for v in listened_to].count(True)
        return sum_sims

    for user in test:
        sessions = data.user_sessions(user)
        user_cos = cosine_similarity(data.u_pref(user).reshape(1, -1), data.m2v_songs)[0]
        user_tn  = data.get_n_largest(user_cos, topN)
        sim_ix   = np.argpartition(users[data.ix_user(user)], -k_sim)[-k_sim:]
        song_sim = np.array([pref(data.ix_user(user), sim_ix, s) for s in songs.index.values])

        for (train_songs, test_songs) in sessions:
            if len(train_songs) > 0:
                c_pref  = data.c_pref(train_songs)

                con_cos = cosine_similarity(c_pref.reshape(1, -1), data.sm2v_songs)[0]

                f_cos   = np.sum([user_cos, con_cos], axis=0)
                UK_cos  = np.sum([song_sim, con_cos], axis=0)
                cos_tn  = data.get_n_largest(con_cos, topN)
                both_tn = data.get_n_largest(f_cos, topN)
                uk_tn   = data.get_n_largest(UK_cos, topN)

                m2vTN.append(get_metrics(user_tn, test_songs))
                sm2vTN.append(get_metrics(cos_tn, test_songs))
                csm2vTN.append(get_metrics(both_tn, test_songs))
                csm2vUK.append(get_metrics(uk_tn, test_songs))

    m_m2vTN     = np.mean(m2vTN, axis=0).tolist()
    m_sm2vTN    = np.mean(sm2vTN, axis=0).tolist()
    m_csm2vTN   = np.mean(csm2vTN, axis=0).tolist()
    m_csm2vUK   = np.mean(csm2vUK, axis=0).tolist()

    return (m_m2vTN, m_sm2vTN, m_csm2vTN, m_csm2vUK)
