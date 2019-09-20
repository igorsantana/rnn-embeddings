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
    
def window_sequences(sequences, window_size):
    w_seq = []
    for seq in sequences:
        seq = seq.split(' ')
        if len(seq) == window_size:
            w_seq.append(seq)
        if len(seq) > window_size:
            for i in range(0, (len(seq) - window_size) + 1):
                w_seq.append(seq[i:i+window_size])
        if len(seq) < window_size:
            seq += ['-'] * (window_size - len(seq))
            w_seq.append(seq)
    return [' '.join(seq) for seq in w_seq]

def gen_seq_files(df, pwd):
    contextual_sessions = df.groupby('session')['song'].agg(list).values
    user_sessions       = df.groupby('user')['song'].agg(list).values
    fc = open(pwd + 'contextual_seqs.txt', 'w+')
    fu = open(pwd + 'listening_seqs.txt', 'w+')
    for session in contextual_sessions:
        fc.write(' '.join(session) + '\n')
    fc.close()
    for listening in user_sessions:
        fu.write(' '.join(listening) + '\n')
    fu.close()

def preprocess(conf):
    ds       = conf['evaluation']['dataset']
    interval = conf['session']['interval']
    if path.exists('dataset/{}/session_listening_history.csv'.format(ds)):
        logging.info('The "%s" dataset is already sessionized', ds)
        return
    logging.info('Started to sessionize dataset "%s"', ds)
    sessionize_user(ds, interval, 'dataset/{}/session_listening_history.csv'.format(ds))
    

