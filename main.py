import re
import os
import yaml
import logging
import argparse
import pandas                               as pd
import numpy                                as np
import multiprocessing                      as mp
import project.evaluation.run               as r
from    datetime                    import datetime
from    project.data.preprocess     import preprocess
from    project.models.embeddings   import embeddings, embeddings_opt
from    project.evaluation.run      import cross_validation


if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser(description='Masters algorithms')
    parser.add_argument('--config', help='Configuration file', type=str)
    args = parser.parse_args()
    conf = yaml.safe_load(open(args.config))
    logging.basicConfig(level=logging.INFO, filename=conf['logfile'], filemode='w', format='%(asctime)s - %(message)s', datefmt='%d/%b/%Y %H:%M:%S')
    logging.info('Config file %s read', args.config)
    logging.info('Project running in (%s) mode', 'opt' if conf['embeddings-opt'] else 'embeddings')
    logging.info('Pre-process started for dataset "%s"', conf['evaluation']['dataset'])
    preprocess(conf)

    if conf['embeddings-opt']:
        methods = embeddings_opt(conf)
    else:
        methods = embeddings(conf)

    cross_validation(conf, methods)



