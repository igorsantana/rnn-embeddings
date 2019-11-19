import sys
import logging
import yaml
import numpy                            as np
import pandas                           as pd
import project.evaluation.metrics       as m
from os.path                        	import exists
from project.data.preparation			import prepare_data, get_embeddings
from project.recsys.helper              import Helper
from datetime                           import datetime
from project.recsys.algorithms        	import execute_algo
import csv
from project.evaluation.ResultReport	import Results

def cross_validation(df, conf, setups):
	params 			= conf['evaluation']
	logging.info('Prepared data for the crossvalidation')
	kfold			= prepare_data(df, conf)
	dataset 		= params['dataset']
	topN			= int(params['topN'])
	k				= int(params['k'])
	results 		= Results(setups, k)
	for setup in setups:
		_, params, path	= setup

		songs		= df['song'].unique().tolist()
		m2v, sm2v   = get_embeddings(path, songs)
		songs       = pd.DataFrame({ 'm2v': [m2v[x] for x in songs], 'sm2v': [sm2v[x] for x in songs]}, index=songs, columns=['m2v','sm2v'])
		
		for train, test in kfold:
			helper 	= Helper(train, test, songs, dataset)
			m2vTN, sm2vTN, csm2vTN, csm2vUK = execute_algo(train.index, test.index, songs, topN, k, helper)
			results.fold_results(params, m2vTN, sm2vTN, csm2vTN, csm2vUK)

		results.finish(params)


