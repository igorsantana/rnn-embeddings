from os import path
import csv
import math
import json
import yaml

import numpy 	as np
import pandas   as pd
import multiprocessing as mp
from datetime import datetime, timedelta

def remove_sessions(df, leq=1):
    group   = df.groupby(by='session').agg(list)
    group   = group['song'].apply(len)
    to_stay = group[group > leq].index.values
    return df[df.session.isin(to_stay)]


def sessionize_user(ds, session_time, s_path):
    df              = pd.read_csv('dataset/{}/listening_history.csv'.format(ds), sep = ',')
    df['timestamp'] = df['timestamp'].astype('datetime64')
    df['dif']       = df['timestamp'].diff()
    df['session']   = df.apply(lambda x: 'NEW_SESSION' if x.dif >= timedelta(minutes=session_time) else 'SAME_SESSION', axis=1)
    s_no = 0
    l_u  = ''
    f = open(s_path, 'w+')
    print(','.join(['user', 'song', 'timestamp', 'session']), file=f)
    print('Sessionized "%s" data file: %s' % (ds, s_path))
    for row in df.values:
        if s_no == 0:
            l_u = row[0]
        if (row[4] == 'NEW_SESSION' and l_u  == row[0]) or (l_u  != row[0]):
            s_no+=1
        row[3] = 's{}'.format(s_no)
        l_u = row[0]
        row[2] = str(row[2])
        print(','.join(row[:-1]), file=f)

def gen_seq_files(df, pwd, window_size):
    c_sessions = df.groupby('session')['song'].agg(list)
    u_sessions = df.groupby('user')['song'].agg(list)
    num_w      = window_size // 2
    fc         = open(pwd + 'c_seqs.csv', 'w+')
    fu         = open(pwd + 'u_seqs.csv', 'w+')
    dict_song  = {}
    for session in c_sessions:
        for ix in range(len(session)):
            b4 = list(range(ix - num_w, ix))
            af = list(range(ix + 1, ix + num_w + 1))
            b4 = [session[i] if i >= 0 else '-' for i in b4]
            af = [session[i] if i < len(session) else '-' for i in af]
            if session[ix] not in dict_song:
                dict_song[session[ix]] = []
            dict_song[session[ix]].append(b4 + [session[ix]] + af)
    for song, values in dict_song.items():
        for seq in values:
            print(song + '\t'+ '{}'.format(seq), file=fc)

    dict_song  = {}
    for session in u_sessions:
        for ix in range(len(session)):
            b4 = list(range(ix - num_w, ix))
            af = list(range(ix + 1, ix + num_w + 1))
            b4 = [session[i] if i >= 0 else '-' for i in b4]
            af = [session[i] if i < len(session) else '-' for i in af]
            if session[ix] not in dict_song:
                dict_song[session[ix]] = []
            dict_song[session[ix]].append(b4 + [session[ix]] + af)
    for song, values in dict_song.items():
        for seq in values:
            print(song + '\t'+ '{}'.format(seq), file=fu)


def preprocess(conf):
    ds       = conf['evaluation']['dataset']
    interval = conf['session']['interval']
    if path.exists('dataset/{}/session_listening_history.csv'.format(ds)):
        print('The "%s" dataset is already sessionized' % ds)
        return
    print('Started to sessionize dataset "%s"' % ds)
    sessionize_user(ds, interval, 'dataset/{}/session_listening_history.csv'.format(ds))
    

