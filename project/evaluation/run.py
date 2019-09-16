import sys
import logging
import yaml
import numpy                            as np
import pandas                           as pd
import project.recsys.algorithms        as runner
import project.data.preparation         as prep
import project.evaluation.metrics       as m

from project.recsys.matrix              import Matrixes
from datetime                           import datetime




def __execute_fold(users, songs, fold, topN, k, ds, file):
	m               = Matrixes(users, songs, ds)
	runner.execute_algo('m2vTN',   users, songs, fold, topN, k, m, file)
	runner.execute_algo('sm2vTN',  users, songs, fold, topN, k, m, file)
	runner.execute_algo('csm2vTN', users, songs, fold, topN, k, m, file)
	runner.execute_algo('csm2vUK', users, songs, fold, topN, k, m, file)
	
def execute_cv(conf, file, embeddings):    
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	topN                    = int(conf['topN'])
	k                       = int(conf['k'])
	df                      = pd.read_csv('dataset/{}/session_listening_history.csv'.format(conf['dataset']))
	cv                      = int(conf['cross-validation'])
	users, songs            = prep.split(df, cv, embeddings, conf['dataset'])
	format 		    		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
	printlog                = lambda x: print(format(x), file=open(file, 'a'))
	
	printlog('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format('Algo','Fold','Prec','Rec', 'F1'))

	for i in range(cv):
			__execute_fold(users, songs, i, topN, k, conf['dataset'], file)
	




