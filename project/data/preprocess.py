import os
import csv
import math
import json
import yaml
import logging
import numpy 	as np
import pandas   as pd
import multiprocessing as mp
from datetime import datetime, timedelta

conf        = yaml.safe_load(open('config.yml'))
fmt_t       = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M')
format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x), file=open(conf['logfile'], 'a'))

def percentage(part, whole):
  return 100 * float(part)/float(whole)

def sessionize_user(df, session_time, s_path):
    df['timestamp'] = df['timestamp'].astype('datetime64')
    df['dif']       = df.timestamp.diff()
    df['session']   = df.apply(lambda x: 'NEW_SESSION' if x.dif >= timedelta(minutes=session_time) else 'SAME_SESSION', axis=1)
    s_no = 0
    l_u  = ''
    f = open(s_path, 'w+')
    print(','.join(['user', 'song', 'timestamp', 'session']), file=f)
    print('Criando sessões no arquivo {}'.format(s_path))
    for row in df.values:
        if s_no == 0:
            l_u = row[0]
        if (row[4] == 'NEW_SESSION' and l_u  == row[0]) or (l_u  != row[0]):
            s_no+=1
        row[3] = 's{}'.format(s_no)
        l_u = row[0]
        row[2] = str(row[2])
        print(','.join(row[:-1]), file=f)
    


def preprocess(dataset, t_session):
    if os.path.exists('dataset/{}/session_listening_history.csv'.format(dataset)):
        printlog('The dataset {} is already sessionized, preprocessing is done.'.format(dataset))
        return
    print('Start to preprocess the dataset.')
    df		= pd.read_csv('dataset/{}/listening_history.csv'.format(dataset), sep = ',')
    logging.info(format('Starting to generate the sessionized dataset'))
    sessionize_user(df, t_session, 'dataset/{}/sublistening_history_40.csv'.format(dataset))
    logging.info(format('Done sessionizing the dataset.'))
    

