import sys
import csv
import os
import yaml
import pickle
import numpy                            as np
import pandas                           as pd
import project.evaluation.metrics       as m
from os.path                        	import exists
from project.data.preparation			import prepare_data, get_embeddings
from project.recsys.helper              import Helper
from datetime                           import datetime
from project.recsys.algorithms        	import execute_algo
from project.evaluation.ResultReport	import Results
from keras.models                   	import model_from_yaml

def get_rnn():
    model = model_from_yaml(open('training_model.yaml','r'))
    model.load_weights('training_weights.h5')
    return model

def skip_all(executed, params, k):
	folds = executed[executed['params'] == params]['folds']
	return folds.max() == k

def skip_fold(executed, params, fold):
	folds = executed[executed['params'] == params]['folds']
	return folds.max() >= fold

def cross_validation(df, conf, setups):
	params 			= conf['evaluation']
	r_paths			= conf['results']
	
	kfold			= prepare_data(df, conf)
	dataset 		= params['dataset']
	topN			= int(params['topN'])
	k				= int(params['k'])
	results 		= Results(setups, k)
	exec_path		= r_paths['full']
	pwd_rec 		= 'tmp/{}/rec/'.format(dataset)

	if not exists(pwd_rec):
		os.mkdir(pwd_rec)
	if not exists(exec_path):
		pd.DataFrame({},columns=['params','algo','folds','prec','rec','f1','map','ndcg@5','p@5']).to_csv(exec_path,index=None,sep='\t')

	executed = pd.read_csv(exec_path, sep='\t')

	for setup in setups:
		_, params, path	= setup
		if not exists(pwd_rec + params):
			os.mkdir(pwd_rec + params)
		if skip_all(executed, params, k):
			continue
		songs		= df['song'].unique().tolist()
		m2v, sm2v   = get_embeddings(path, songs)
		songs       = pd.DataFrame({ 'm2v': [m2v[x] for x in songs], 'sm2v': [sm2v[x] for x in songs]}, index=songs, columns=['m2v','sm2v'])
		fold = 1
		for train, test in kfold:
			if skip_fold(executed, params, fold):
				fold+=1
				continue
			time = datetime.now().strftime('%d/%m/%Y %H:%M')
			print('%s | fold-%d | Running recsys w/ k-fold with the following params: %s' % (time, fold, params))
			helper 	= Helper(train, test, songs, dataset)
			m2vTN, sm2vTN, csm2vTN, csm2vUK = execute_algo(train.index, test.index, songs, topN, k, helper, pwd_rec + params)
			res 							= results.fold_results(params, m2vTN, sm2vTN, csm2vTN, csm2vUK, fold)
			res.to_csv(exec_path, sep='\t', mode='a', index=None, header=None)
			fold+=1
