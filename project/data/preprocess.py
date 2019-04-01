import os
import csv
import math
import json
import logging
import numpy 	as np
import pandas   as pd
import multiprocessing as mp
from datetime import datetime

fmt_t       = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M')
format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x))

def percentage(part, whole):
  return 100 * float(part)/float(whole)

def sessionize_user(df, session_time, writer):
    s_counter = 0
    for row in df.itertuples(index=True):
        idx = getattr(row, 'Index')
        if idx != 0:
            dif = fmt_t(getattr(df.iloc[idx - 1], 'timestamp')) - fmt_t(getattr(row, 'timestamp'))  
            c1  = abs(int(round(dif.total_seconds() / 60))) >  session_time
            c2  = (getattr(df.iloc[idx - 1], 'user') != getattr(row, 'user'))
            if c1 or c2: s_counter+=1
        writer.writerow([getattr(row, 'user'), getattr(row, 'song'), getattr(row, 'timestamp'), s_counter])
        print('Sessionizing the dataset: {}%'.format(round(percentage(idx, len(df.index)),2)), end='\r', flush=True)




def preprocess(dataset, t_session):
    printlog('Checking if the dataset {} is already sessionized.'.format(dataset))
    if os.path.exists('dataset/{}/session_listening_history.csv'.format(dataset)):
        printlog('The dataset {} is already sessionized, preprocessing is done.'.format(dataset))
        return
    else: 
        print('Start to preprocess the dataset.')
    logging.info(format('Started reading the dataset'))
    df		= pd.read_csv('dataset/{}/listening_history.csv'.format(dataset), sep = ',')
    logging.info(format('The dataset has been read'))
    logging.info(format('Starting to generate the sessionized dataset'))
    f       = csv.writer(open('dataset/{}/session_listening_history.csv'.format(dataset),'w'))
    f.writerow(['user','song','timestamp', 'session'])
    sessionize_user(df, t_session, f)
    logging.info(format('Done sessionizing the dataset.'))
    

