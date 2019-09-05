import sys
import logging
import yaml
import numpy                            as np
import pandas                           as pd
import project.recsys.algorithms        as runner
import project.data.preparation         as prep
import project.evaluation.metrics       as m
from gensim.models                      import Word2Vec, Doc2Vec
from project.recsys.matrix              import Matrixes
from datetime                           import datetime
def __load_models(dataset, is_doc):
	if is_doc:
		return Doc2Vec.load('tmp/{}/models/doc2vec.model'.format(dataset)), Doc2Vec.load('tmp/{}/models/sessiondoc2vec.model'.format(dataset))
	return Word2Vec.load('tmp/{}/models/music2vec.model'.format(dataset)), Word2Vec.load('tmp/{}/models/sessionmusic2vec.model'.format(dataset))

def __execute_fold(users, songs, fold, topN, k, ds, is_doc, m2v, sm2v):
	m               = Matrixes(users, songs, ds, is_doc, m2v, sm2v)
	runner.execute_algo('m2vTN',   users, songs, fold, topN, k, m)
	runner.execute_algo('sm2vTN',  users, songs, fold, topN, k, m)
	runner.execute_algo('csm2vTN', users, songs, fold, topN, k, m)
	runner.execute_algo('csm2vUK', users, songs, fold, topN, k, m)
	
def execute_cv(conf, file, is_doc):    
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	topN                    = int(conf['topN'])
	k                       = int(conf['k'])
	m2v, sm2v               = __load_models(conf['dataset'], is_doc)
	df                      = pd.read_csv('dataset/{}/session_listening_history_reduzido.csv'.format(conf['dataset']))
	cv                      = int(conf['cross-validation'])
	users, songs            = prep.split(df, cv, m2v, sm2v)
	format 		    		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
	printlog                = lambda x: print(format(x), file=open(file, 'a'))
	
	printlog('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format('Algo','Fold','Prec','Rec', 'F1'))

	for i in range(cv):
			__execute_fold(users, songs, i, topN, k, conf['dataset'], is_doc, m2v, sm2v)
	




