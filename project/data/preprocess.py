from os import path
import csv
import math
import json
import yaml
import logging
import numpy 	as np
import pandas   as pd
import multiprocessing as mp
from datetime import datetime, timedelta

def sessionize_user(ds, session_time, s_path):
    df              = pd.read_csv('dataset/{}/listening_history.csv'.format(ds), sep = ',')
    df['timestamp'] = df['timestamp'].astype('datetime64')
    df['dif']       = df.timestamp.diff()
    df['session']   = df.apply(lambda x: 'NEW_SESSION' if x.dif >= timedelta(minutes=session_time) else 'SAME_SESSION', axis=1)
    s_no = 0
    l_u  = ''
    f = open(s_path, 'w+')
    print(','.join(['user', 'song', 'timestamp', 'session']), file=f)
    logging.info('Sessionized "%s" data file: %s', ds, s_path)
    for row in df.values:
        if s_no == 0:
            l_u = row[0]
        if (row[4] == 'NEW_SESSION' and l_u  == row[0]) or (l_u  != row[0]):
            s_no+=1
        row[3] = 's{}'.format(s_no)
        l_u = row[0]
        row[2] = str(row[2])
        print(','.join(row[:-1]), file=f)

def window_sequences(song, seqs, w_size):
    w = w_size // 2
    to_r = []
    for seq in seqs:
        seq = np.array(seq)
        ixs = np.where(song == seq)[0]
        seql = seq.tolist()
        for ix in ixs:
            if ix - w < 0:
                ab = abs(ix - w)
                b4 = ['-'] * ab + (seql[0:ix])
            else:
                b4 = seql[ix - w:ix]
            if ix + w > len(seq) - 1:
                ab = abs(ix + w) - (len(seq) - 1)
                af = seql[ix + 1: len(seq)] + (['-'] * ab )
            else:
                af = seql[ix + 1: ix + w + 1]
            to_r.append(np.array(b4 + [seql[ix]] + af))
    return np.array(to_r)


def gen_seq_files(df, pwd, window_size):
    c_sessions = df.groupby('session')['song'].agg(list)
    u_sessions = df.groupby('user')['song'].agg(list)
    c_sessions = c_sessions[c_sessions.apply(len) > 1].values
    u_sessions = u_sessions[u_sessions.apply(len) > 1].values
    songs      = df.song.unique()
    fc         = open(pwd + 'c_seqs.csv', 'w+')
    fu         = open(pwd + 'u_seqs.csv', 'w+')
    for song in songs:
        c_seqs = window_sequences(song, c_sessions, window_size)
        u_seqs = window_sequences(song, u_sessions, window_size)
        for seq in c_seqs:
            print(song + '\t'+ '[{}]'.format( ','.join(seq)), file=fc)
        for seq in u_seqs:
            print(song + '\t'+ '[{}]'.format( ','.join(seq)), file=fu)
    




def preprocess(conf):
    ds       = conf['evaluation']['dataset']
    interval = conf['session']['interval']
    if path.exists('dataset/{}/session_listening_history.csv'.format(ds)):
        logging.info('The "%s" dataset is already sessionized', ds)
        return
    logging.info('Started to sessionize dataset "%s"', ds)
    sessionize_user(ds, interval, 'dataset/{}/session_listening_history.csv'.format(ds))
    

