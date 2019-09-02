import os
import sys
import time
import math
import logging
import numba
import timeit
import itertools
import statistics                           as s
import multiprocessing                      as mp
import numpy                                as np
import pandas                               as pd
import project.evaluation.metrics           as m
from scipy.spatial.distance                 import cdist, cosine, squareform, pdist
from datetime                               import datetime
from operator                               import add
from functools                              import partial
from numpy                                  import dot
from numpy.linalg                           import norm
from sklearn.metrics.pairwise               import cosine_similarity


format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x), file=open('output.log', 'a'))
st_time     = lambda _: time.time()
f_time      = lambda st: time.time() - st
train       = lambda session: session[:len(session)//2]
test        = lambda session: session[len(session)//2:]

def run_m2vTN(users,songs, fold, topN):
    printlog('[Fold {}] Started to evaluate users for the m2vTN algorithm'.format(fold))
    test_users      = users[users['tt_{}'.format(fold)] == 'test'].index.tolist()
    song_m2v_vecs   = np.array(songs['m2v'].tolist(), dtype= np.float64)
    user_metrics    = []
    for user in test_users:
        sessions        = [session for session in users.loc[user, 'history'] if len(session) > 1]
        train_sessions  = list(map(train, sessions))
        test_sessions   = list(map(test, sessions))
        flat_songs      = sum(train_sessions, [])
        flat_vecs       = songs.loc[flat_songs, 'm2v'].tolist()
        user_pref       = np.mean(np.array(flat_vecs), axis=0)
        songs['cos']    = cosine_similarity([user_pref], song_m2v_vecs)[0]
        top_n           = songs.nlargest(topN, 'cos').index.tolist()
        metrics         = []
        for test_session in test_sessions:
            p                   = m.Precision(top_n, test_session)
            r                   = m.Recall(top_n, test_session)
            fm                  = m.FMeasure(p, r)
            metrics.append([p, r, fm])
        user_metrics.extend(metrics)
    means = np.mean(user_metrics, axis=0)
    printlog('Precision: ' + str(means[0]))
    printlog('Recall: ' + str(means[1]))
    printlog('FMeasure: ' + str(means[2]))
    printlog('[Fold {}] Finished to evaluate users for the m2vTN algorithm'.format(fold))
    return means


def run_sm2vTN(users,songs, fold, topN):
    printlog('[Fold {}] Started to evaluate users for the sm2vTN algorithm'.format(fold))
    test_users      = users[users['tt_{}'.format(fold)] == 'test'].index.tolist()
    song_sm2v_vecs  = np.array(songs['sm2v'].tolist(), dtype= np.float64)
    user_metrics    = []
    for user in test_users:
        sessions        = [session for session in users.loc[user, 'history'] if len(session) > 1]
        metrics         = []
        for session in sessions:
            train_session   = train(session)
            test_session    = test(session)
            flat_vecs       = songs.loc[train_session, 'sm2v'].tolist()
            context_pref    = np.mean(np.array(flat_vecs), axis=0)
            songs['cos']    = cosine_similarity([context_pref], song_sm2v_vecs)[0]
            top_n           = songs.nlargest(topN, 'cos').index.tolist()
            p               = m.Precision(top_n, test_session)
            r               = m.Recall(top_n, test_session)
            fm              = m.FMeasure(p, r)
            metrics.append([p, r, fm])
        user_metrics.extend(metrics)
    means = np.mean(user_metrics, axis=0)
    printlog('Precision: ' + str(means[0]))
    printlog('Recall: ' + str(means[1]))
    printlog('FMeasure: ' + str(means[2]))
    printlog('[Fold {}] Finished to evaluate users for the sm2vTN algorithm'.format(fold))
    return means

def run_csm2vTN(users,songs, fold, topN):
    printlog('[Fold {}] Started to evaluate users for the csm2vTN algorithm'.format(fold))
    test_users      = users[users['tt_{}'.format(fold)] == 'test'].index.tolist()
    song_m2v_vecs   = np.array(songs['m2v'].tolist(), dtype= np.float64)
    song_sm2v_vecs  = np.array(songs['sm2v'].tolist(), dtype= np.float64)
    user_metrics    = []
    for user in test_users:
        sessions            = [session for session in users.loc[user, 'history'] if len(session) > 1]
        train_sessions      = list(map(train, sessions))
        flat_songs          = sum(train_sessions, [])
        m2v_vecs            = songs.loc[flat_songs, 'm2v'].tolist()
        user_pref           = np.mean(np.array(m2v_vecs), axis=0)
        songs['user_cos']   = cosine_similarity([user_pref], song_m2v_vecs)[0]
        metrics             = []
        for session in sessions:
            train_data              = train(session)
            test_data               = test(session)
            sm2v_vecs               = songs.loc[train_data, 'sm2v'].tolist()
            context_pref            = np.mean(np.array(sm2v_vecs), axis=0)
            songs['context_cos']    = cosine_similarity([context_pref], song_sm2v_vecs)[0]
            songs['cos']            = songs['user_cos'] + songs['context_cos']
            top_n                   = songs.nlargest(topN, 'cos').index.tolist()
            p                       = m.Precision(top_n, test_data)
            r                       = m.Recall(top_n, test_data)
            fm                      = m.FMeasure(p, r)
            metrics.append([p, r, fm])
        user_metrics.extend(metrics)
    means = np.mean(user_metrics, axis=0)
    printlog('Precision: ' + str(means[0]))
    printlog('Recall: ' + str(means[1]))
    printlog('FMeasure: ' + str(means[2]))
    printlog('[Fold {}] Finished to evaluate users for the csm2vTN algorithm'.format(fold))
    return means


def run_csm2vUK(users, songs, fold, topN, k, ds):
    printlog('[Fold {}] Started to evaluate users for the csm2vUK algorithm'.format(fold))
    def u_pref(user):
        history      = users[users.index == user]['history'].values.tolist()[0]
        flat_history = [song for session in history for song in session]
        unique_songs = list(set(flat_history))
        flat_history = [songs.loc[song, 'm2v'] for song in flat_history]
        return np.mean(flat_history, axis=0), unique_songs

        
    def pref(u, k_similar, song):
        listened_to = [(k, matrix_u_songs[k, songs_ix[song]] == 1) for k in k_similar]
        sum_sims = 0
        for u_k, listen in listened_to:
            if listen == True:
                sum_sims += matrix_users[u][u_k] / [v[1] for v in listened_to].count(True)
        return sum_sims
                
    user_metrics    = []
    all_songs       = songs.index.values
    ix_users        = { v:k for k,v in enumerate(users.index) }
    songs_ix        = { v:k for k,v in enumerate(all_songs) }

    ix_pref         = { v:u_pref(k)[0] for (k,v) in ix_users.items() }
    ix_u_songs      = { v:u_pref(k)[1] for (k,v) in ix_users.items() }

    num_users       = len(users.index)
    num_songs       = len(all_songs)

    matrix_users    = np.zeros((num_users, num_users))
    matrix_u_songs  = np.zeros((num_users, len(all_songs)))

    if os.path.isfile('tmp/{}/matrix_users.npy'.format(ds)):
        print('Reading')
        matrix_users = np.load('tmp/{}/matrix_users.npy'.format(ds))
    else:
        for ix in range(num_users):
            u_array = np.array([ix_pref[i] for i in range(num_users)])
            y_array = np.zeros(num_users)
            for j in range(num_users):
                y_array[j] = math.sqrt(len(ix_u_songs[ix]) + len(ix_u_songs[j]))
            cos = cosine_similarity(np.array([ix_pref[ix]]), u_array)
            val = np.sum([cos, y_array], axis=0) 
            matrix_users[ix] = np.divide(np.ones(val.shape), val)
        np.save('tmp/{}/matrix_users'.format(ds), matrix_users)
    if os.path.isfile('tmp/{}/matrix_user_songs.npy'.format(ds)):
        print('Reading')
        matrix_users = np.load('tmp/{}/matrix_user_songs.npy'.format(ds))
    else:
        for u in range(num_users):
            songs_ids = [songs_ix[s] for s in ix_u_songs[u]]
            y_array = np.zeros(num_songs)
            for s in songs_ids:
                y_array[s] = 1
            matrix_u_songs[u] = y_array
        np.save('tmp/{}/matrix_user_songs'.format(ds), matrix_u_songs)

    t_users     = users[users['tt_{}'.format(fold)] == 'test'].index

    for user in t_users:
        sim_ix          = np.argpartition(matrix_users[ix_users[user]], -k)[-k:]
        songs_sim       = np.array([pref(ix_users[user], sim_ix, s) for s in all_songs])
        sessions        = [session for session in users.loc[user, 'history'] if len(session) > 1]
        metrics         = []
        for session in sessions:
            train_session   = train(session)
            test_session    = test(session)
            song_vecs       = songs.loc[train_session, 'sm2v']
            c_pref          = np.mean(song_vecs, axis=0)
            cos             = cosine_similarity([c_pref], np.array(songs['sm2v'].values.tolist()))
            sum_v           = np.sum([songs_sim, cos], axis=0)
            songs['pref']   = sum_v[0]
            top_n           = songs.nlargest(topN, 'pref').index.tolist()
            p               = m.Precision(top_n, test_session)
            r               = m.Recall(top_n, test_session)
            fm              = m.FMeasure(p, r)
            metrics.append([p, r, fm])
        user_metrics.extend(metrics)
    means = np.mean(user_metrics, axis=0)
    printlog('Precision: ' + str(means[0]))
    printlog('Recall: ' + str(means[1]))
    printlog('FMeasure: ' + str(means[2]))
    printlog('[Fold {}] Finished to evaluate users for the csm2vUK algorithm'.format(fold))


def execute_algo(name, users, songs, fold, topN, k, ds):
    if name == 'm2vTN':
        return run_m2vTN(users, songs, fold, topN)
    if name == 'sm2vTN':
        return run_sm2vTN(users, songs, fold, topN)
    if name == 'csm2vTN':
        return run_csm2vTN(users, songs, fold, topN)
    if name == 'csm2vUK':        
        return run_csm2vUK(users,songs, fold, topN, k, ds)
    return 0
    
    

@numba.jit(target='cpu', nopython=True, parallel=True)
def cos_matrix(u, M):
    scores = np.zeros(M.shape[0])
    for i in numba.prange(M.shape[0]):
        v = M[i]
        m = u.shape[0]
        udotv = 0
        u_norm = 0
        v_norm = 0
        for j in range(m):
            if (np.isnan(u[j])) or (np.isnan(v[j])):
                continue

            udotv += u[j] * v[j]
            u_norm += u[j] * u[j]
            v_norm += v[j] * v[j]

        u_norm = np.sqrt(u_norm)
        v_norm = np.sqrt(v_norm)

        if (u_norm == 0) or (v_norm == 0):
            ratio = 1.0
        else:
            ratio = udotv / (u_norm * v_norm)
        scores[i] = ratio
    return scores

