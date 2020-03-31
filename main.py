import re
import os
import yaml
import pickle
import argparse
import pandas                               as pd
import numpy                                as np
import multiprocessing                      as mp
import project.evaluation.run               as r
from    os.path                     import exists
from    datetime                    import datetime
from    project.data.preprocess     import preprocess, remove_sessions
from    project.models.embeddings   import embeddings
from    project.evaluation.run      import cross_validation


if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description='RNN Embeddings')
    parser.add_argument('--config', help='Configuration file', type=str)
    args = parser.parse_args()
    conf = yaml.safe_load(open(args.config))
    
    print('The configuration file "%s" was read.' % args.config)
    print('Pre-process started for dataset "%s"' % conf['evaluation']['dataset'])
    
    preprocess(conf)
    
    ds  = conf['evaluation']['dataset']
    df  = pd.read_csv('dataset/{}/session_listening_history.csv'.format(ds), sep = ',')

    emb_path = 'tmp/{}/models/ids.npy'.format(ds)
    
    if not exists(emb_path):
    	embeddings(df, conf)
    ids = np.load(emb_path)
    cross_validation(df, conf, ids)
