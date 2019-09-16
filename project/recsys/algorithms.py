import os
import sys
import time
import yaml
import multiprocessing                      as mp
import numpy                                as np
import project.evaluation.metrics           as m
from datetime                               import datetime
from sklearn.metrics.pairwise               import cosine_similarity

conf        = yaml.safe_load(open('config.yml'))
format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x), file=open(conf['logfile'], 'a'))
st_time     = lambda _: time.time()
f_time      = lambda st: time.time() - st
train       = lambda session: session[:len(session)//2]
test        = lambda session: session[len(session)//2:]


def run_m2vTN(users,songs, fold, topN, mat, file):
    print('[Fold {}] Started to evaluate users for the m2vTN algorithm'.format(fold))

    test_users      = users[users['tt_{}'.format(fold)] == 'test'].index.tolist()
    user_metrics    = []

    for user in test_users:
        sessions        = [session for n_s, session in users.loc[user, 'history'] if len(session) > 1]
        cos             = cosine_similarity(mat.u_pref(user)[0].reshape(1, -1), mat.m2v_songs)[0]
        top_n           = mat.get_n_largest(cos, topN)
        metrics         = []
        for test_session in [test(session) for session in sessions]:
            p                   = m.Precision(top_n, test_session)
            r                   = m.Recall(top_n, test_session)
            fm                  = m.FMeasure(p, r)
            metrics.append([p, r, fm])
        user_metrics.extend(metrics)
    means = np.mean(user_metrics, axis=0)
    print(format('{:^10s}{:^10d}{:^10.5f}{:^10.5f}{:^10.5f}'.format('m2vTN',fold, means[0], means[1], means[2])), file=open(file, 'a'))
    print('[Fold {}] Finished to evaluate users for the m2vTN algorithm'.format(fold))
    


def run_sm2vTN(users,songs, fold, topN, mat, file):
    print('[Fold {}] Started to evaluate users for the sm2vTN algorithm'.format(fold))
    test_users      = users[users['tt_{}'.format(fold)] == 'test'].index.tolist()
    user_metrics    = []
    for user in test_users:
        sessions        = [(n_s, session) for n_s, session in users.loc[user, 'history'] if len(session) > 1]
        metrics         = []
        for n_s, session in sessions:
            test_session    = test(session)
            c_pref          = mat.c_pref(n_s, train(session))
            cos             = cosine_similarity(c_pref.reshape(1, -1), mat.sm2v_songs)[0]
            top_n           = mat.get_n_largest(cos, topN)
            p               = m.Precision(top_n, test_session)
            r               = m.Recall(top_n, test_session)
            fm              = m.FMeasure(p, r)
            metrics.append([p, r, fm])
        user_metrics.extend(metrics)

    means = np.mean(user_metrics, axis=0)
    print(format('{:^10s}{:^10d}{:^10.5f}{:^10.5f}{:^10.5f}'.format('sm2vTN',fold, means[0], means[1], means[2])), file=open(file, 'a'))
    print('[Fold {}] Finished to evaluate users for the sm2vTN algorithm'.format(fold))
    return means

def run_csm2vTN(users,songs, fold, topN, mat, file):
    print('[Fold {}] Started to evaluate users for the csm2vTN algorithm'.format(fold))
    test_users      = users[users['tt_{}'.format(fold)] == 'test'].index.tolist()
    user_metrics    = []
    for user in test_users:
        sessions            = [(n_s, session) for n_s, session in users.loc[user, 'history'] if len(session) > 1]
        u_pref              = mat.u_pref(user)[0]
        cos                 = cosine_similarity(u_pref.reshape(1, -1), mat.m2v_songs)[0]
        metrics             = []
        for (n_s, session) in sessions:
            test_session            = test(session)
            c_pref                  = mat.c_pref(n_s, train(session))
            c_cos                   = cosine_similarity(c_pref.reshape(1, -1), mat.m2v_songs)[0]
            f_cos                   = np.sum([cos, c_cos], axis=0)
            top_n                   = mat.get_n_largest(f_cos, topN)
            p                       = m.Precision(top_n, test_session)
            r                       = m.Recall(top_n, test_session)
            fm                      = m.FMeasure(p, r)
            metrics.append([p, r, fm])
        user_metrics.extend(metrics)
    means = np.mean(user_metrics, axis=0)
    print(format('{:^10s}{:^10d}{:^10.5f}{:^10.5f}{:^10.5f}'.format('csm2vTN',fold, means[0], means[1], means[2])), file=open(file, 'a'))
    print('[Fold {}] Finished to evaluate users for the csm2vTN algorithm'.format(fold))
    return means


def run_csm2vUK(users, songs, fold, topN, k, mat, file):
    print('[Fold {}] Started to evaluate users for the csm2vUK algorithm'.format(fold))

    matrix_u_songs  = mat.us_matrix()
    matrix_users    = mat.uu_matrix()
    user_metrics    = []
    all_songs       = songs.index.values
    t_users         = users[users['tt_{}'.format(fold)] == 'test'].index

    def pref(u, k_similar, song):
        listened_to = [(k, matrix_u_songs[k, mat.song_ix(song)] == 1) for k in k_similar]
        sum_sims = 0
        for u_k, listen in listened_to:
            if listen == True:
                sum_sims += matrix_users[u][u_k] / [v[1] for v in listened_to].count(True)
        return sum_sims

    for user in t_users:
        sim_ix          = np.argpartition(matrix_users[mat.ix_user(user)], -k)[-k:]
        songs_sim       = np.array([pref(mat.ix_user(user), sim_ix, s) for s in all_songs])
        sessions        = [(n_s, session) for n_s, session in users.loc[user, 'history'] if len(session) > 1]
        metrics         = []
        for (n_s, session) in sessions:
            test_session    = test(session)
            c_pref          = mat.c_pref(n_s, train(session))
            cos             = cosine_similarity(c_pref.reshape(1, -1), mat.sm2v_songs)[0]
            sum_v           = np.sum([songs_sim, cos], axis=0)
            top_n           = mat.get_n_largest(sum_v, topN)
            p               = m.Precision(top_n, test_session)
            r               = m.Recall(top_n, test_session)
            fm              = m.FMeasure(p, r)
            metrics.append([p, r, fm])
        user_metrics.extend(metrics)
    means = np.mean(user_metrics, axis=0)
    print(format('{:^10s}{:^10d}{:^10.5f}{:^10.5f}{:^10.5f}'.format('csm2vUK',fold, means[0], means[1], means[2])), file=open(file, 'a'))
    print('[Fold {}] Finished to evaluate users for the csm2vUK algorithm'.format(fold))


def execute_algo(name, users, songs, fold, topN, k, m, file):
    if name == 'm2vTN':
        return run_m2vTN(users, songs, fold, topN, m, file)
    if name == 'sm2vTN':
        return run_sm2vTN(users, songs, fold, topN, m, file)
    if name == 'csm2vTN':
        return run_csm2vTN(users, songs, fold, topN, m, file)
    if name == 'csm2vUK':        
        return run_csm2vUK(users,songs, fold, topN, k, m, file)
    return 0
