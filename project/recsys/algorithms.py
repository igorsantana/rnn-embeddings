import os
import sys
import time
import yaml
import pickle
import multiprocessing                      as mp
import numpy                                as np
from project.evaluation.metrics             import get_metrics
from datetime                               import datetime
from sklearn.metrics.pairwise               import cosine_similarity


def write_rec(pwd, sessions):
    f = open(pwd, 'wb')
    pickle.dump(sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def recs(session ,original, mtn_rec, smtn_rec, csmtn_rec, csmuk_rec):
    return ({ 'session': session, 'original': original, 'mtn_rec': mtn_rec.tolist(), 'smtn_rec': smtn_rec.tolist(), 'csmtn_rec': csmtn_rec.tolist(),  'csmuk_rec': csmtn_rec.tolist()})


def rnn_get_topN(train_songs, topN, rnn, song2ix):
    recommendations = []
    ix2song         = {v: k for k, v in song2ix.items()}
    train_songs_ix  = [song2ix[songs] for songs in train_songs]
    while len(recommendations) < topN:
        rec = rnn.predict(np.array([train_songs_ix[-5:]]), batch_size=1)
        value = np.argmax(rec)
        recommendations.append(value)
        train_songs_ix.append(value)

    return [ix2song[song] for song in recommendations]


def execute_algo(train, test, songs, topN, k_sim, data, pwd, rnn, song2ix):

    m2vTN   = []
    sm2vTN  = []
    csm2vTN = [] 
    csm2vUK = []
    rnnTN   = []
    
    u_songs  = data.us_matrix()
    users    = data.uu_matrix()


    def report_users(num_users):
        def f_aux(ix_user, user_id, algo):
            return '[{}/{}] Running algorithm {} for user {}!'.format(ix_user, num_users,algo, user_id)
        return f_aux

    num_users   = len(test)
    rep         = report_users(num_users)
    u           = 1

    def pref(u, k_similar, song):
        listened_to = [(k, u_songs[k, data.song_ix(song)] == 1) for k in k_similar]
        sum_sims = 0
        for u_k, listen in listened_to:
            if listen == True:
                sum_sims += users[u][u_k] / [v[1] for v in listened_to].count(True)
        return sum_sims


    for user in test:
        sessions = data.user_sessions(user)

        user_cos = cosine_similarity(data.u_pref(user).reshape(1, -1), data.m2v_songs, )[0]
        user_tn  = data.get_n_largest(user_cos, topN)

        f = open(pwd + '/' + user.replace('/', '_'), 'wb')
        pickle.dump({}, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        print(rep(u, user, 'M-TN'), flush=False, end='\r')

        sim_ix   = np.argpartition(users[data.ix_user(user)], -k_sim)[-k_sim:]
        song_sim = np.array([pref(data.ix_user(user), sim_ix, s) for s in songs.index.values])
        to_write = []
        s = 1
        for (train_songs, test_songs) in sessions:
            if len(train_songs) > 0:
                c_pref  = data.c_pref(train_songs)
                # print(rep(u, user, 'RNN-TN'), flush=False, end='\r')
                # rnn_tn  = rnn_get_topN(train_songs, topN, rnn, song2ix)

                con_cos = cosine_similarity(c_pref.reshape(1, -1), data.sm2v_songs)[0]
                print(rep(u, user, 'SM-TN'), flush=False, end='\r')
                f_cos   = np.sum([user_cos, con_cos], axis=0)
                print(rep(u, user, 'CSM-TN'), flush=False, end='\r')
                UK_cos  = np.sum([song_sim, con_cos], axis=0)
                print(rep(u, user, 'CSM-UK'), flush=False, end='\r')
                cos_tn  = data.get_n_largest(con_cos, topN)
                
                both_tn = data.get_n_largest(f_cos, topN)
                uk_tn   = data.get_n_largest(UK_cos, topN)
                to_write.append(recs(s, test_songs, user_tn, cos_tn, both_tn, uk_tn))
                m2vTN.append(get_metrics(user_tn, test_songs))
                sm2vTN.append(get_metrics(cos_tn, test_songs))
                csm2vTN.append(get_metrics(both_tn, test_songs))
                csm2vUK.append(get_metrics(uk_tn, test_songs))
                rnnTN.append(get_metrics(cos_tn, test_songs))
                s+=1
        
        write_rec(pwd + '/' + user.replace('/', '_'), to_write)
        u+=1

    m_m2vTN     = np.mean(m2vTN, axis=0).tolist()
    m_sm2vTN    = np.mean(sm2vTN, axis=0).tolist()
    m_csm2vTN   = np.mean(csm2vTN, axis=0).tolist()
    m_csm2vUK   = np.mean(csm2vUK, axis=0).tolist()
    m_rnnTN     = np.mean(rnnTN, axis=0).tolist()
    return (m_m2vTN, m_sm2vTN, m_csm2vTN, m_csm2vUK, m_rnnTN)
