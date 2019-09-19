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
from    project.models.embeddings   import embeddings
from    project.evaluation.run      import cross_validation


if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser(description='Masters algorithms')
    parser.add_argument('--config', help='Configuration file', type=str)
    args = parser.parse_args()
    conf = yaml.safe_load(open(args.config))
    logging.basicConfig(level=logging.INFO, filename=conf['logfile'], filemode='w', format='%(asctime)s - %(message)s', datefmt='%d/%b/%Y %H:%M:%S')
    logging.info('Config file %s read', args.config)
    logging.info('Project running in (%s) mode', 'recsys' if conf['embeddings-only'] else 'embeddings')
    logging.info('Pre-process started for dataset "%s"', conf['evaluation']['dataset'])
    preprocess(conf)
    methods = embeddings(conf)
    
    if conf['embeddings-only']:
        logging.info('Our job here is done, embeddings generated in "tmp/%s" folder.', conf['evaluation']['dataset'])
        exit()

    cross_validation(conf, methods)


    # vals = {}
    # cache = []
    # i = 0
    # for line in reversed(list(open(conf['logfile']))):
    #     l = line.rstrip()
    #     if 'Algo' not in l: 
    #         cache.append(l)
    #     else:
    #         vals[i] = cache
    #         cache = []
    #         i+=1
    # regex = re.compile(".*?\[(.*?)\]")
    # for key in vals.keys():
    #     l = vals[key]
    #     l = [re.sub(regex, '', x) for x in l]
    #     l = [x.split(' ') for x in l if 'The' not in x]
    #     l = [ [x for x in strings if x] for strings in l]
    #     df = pd.DataFrame(l)
    #     df.columns = ['algo', 'fold', 'rec', 'prec', 'f1']
    #     df = df.astype({ 'fold': 'int32', 'rec': 'float32', 'prec': 'float32', 'f1': 'float32'})
    #     print(df.groupby(by='algo').mean().drop(columns='fold'))
    



