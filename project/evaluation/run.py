import sys
import logging
import yaml
import numpy                            as np
import pandas                           as pd
import project.evaluation.metrics       as m
from os.path                        	import exists
from project.data.preparation			import prepare_data, get_embeddings, get_embeddings_opt
from project.recsys.helper              import Helper
from datetime                           import datetime
from project.recsys.algorithms        	import execute_algo
import csv

def report_results(algo, fold, prec, rec, f1, f):
	if isinstance(prec, float):
		logging.info('{:^10s}{:^10s}{:^10.5f}{:^10.5f}{:^10.5f}'.format(algo, str(fold), prec, rec, f1))
	else:
		logging.info('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format(algo, str(fold), prec, rec, f1))
	print(';'.join([str(x) for x in [algo, fold, prec, rec, f1]]), file=f)

def report_results_opt(conf, algo, fold, prec, rec, f1):
	if isinstance(prec, float):
		logging.info('{:^10s}{:^10s}{:^10s}{:^10.5f}{:^10.5f}{:^10.5f}'.format(conf, algo, str(fold), prec, rec, f1))
	else:
		logging.info('{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format(conf, algo, str(fold), prec, rec, f1))
	

def m_app(metrics, fold, m2vTN, sm2vTN, csm2vTN, csm2vUK):
	metrics = metrics.append(pd.Series(['m2vTN', fold] + m2vTN, index=metrics.columns), ignore_index=True)
	metrics = metrics.append(pd.Series(['sm2vTN', fold] + sm2vTN, index=metrics.columns), ignore_index=True)
	metrics = metrics.append(pd.Series(['csm2vTN', fold] + csm2vTN, index=metrics.columns), ignore_index=True)
	metrics = metrics.append(pd.Series(['csm2vUK', fold] + csm2vUK, index=metrics.columns), ignore_index=True)
	return metrics

def m_app_opt(conf,metrics, fold, m2vTN, sm2vTN, csm2vTN, csm2vUK):
	metrics = metrics.append(pd.Series([conf, 'm2vTN', fold] + m2vTN, index=metrics.columns), ignore_index=True)
	metrics = metrics.append(pd.Series([conf, 'sm2vTN', fold] + sm2vTN, index=metrics.columns), ignore_index=True)
	metrics = metrics.append(pd.Series([conf, 'csm2vTN', fold] + csm2vTN, index=metrics.columns), ignore_index=True)
	metrics = metrics.append(pd.Series([conf, 'csm2vUK', fold] + csm2vUK, index=metrics.columns), ignore_index=True)
	return metrics

def report( fold, m2vTN, sm2vTN, csm2vTN, csm2vUK, file):
	report_results('m2vTN', fold, m2vTN[0], m2vTN[1], m2vTN[2], file)
	report_results('sm2vTN', fold, sm2vTN[0], sm2vTN[1], sm2vTN[2], file)
	report_results('csm2vTN', fold, csm2vTN[0], csm2vTN[1], csm2vTN[2], file)
	report_results('csm2vUK', fold, csm2vUK[0], csm2vUK[1], csm2vUK[2], file)

def report_opt(conf, fold, m2vTN, sm2vTN, csm2vTN, csm2vUK):
	report_results_opt(conf, 'm2vTN', fold, m2vTN[0], m2vTN[1], m2vTN[2])
	report_results_opt(conf, 'sm2vTN', fold, sm2vTN[0], sm2vTN[1], sm2vTN[2])
	report_results_opt(conf, 'csm2vTN', fold, csm2vTN[0], csm2vTN[1], csm2vTN[2])
	report_results_opt(conf, 'csm2vUK', fold, csm2vUK[0], csm2vUK[1], csm2vUK[2])

def cross_validation(conf, methods):
	params 			= conf['evaluation']
	df				= pd.read_csv('dataset/{}/session_listening_history.csv'.format(params['dataset']))
	logger          = logging.getLogger()
	logging.info('Prepared data for the crossvalidation')
	kfold	= prepare_data(df, conf)
	if conf['embeddings-opt']:
		f = open('tmp/{}/results.csv'.format(params['dataset']), 'w+')
		w = csv.writer(f)
		for id in methods.keys():
			logging.info('Running the recommender systems with embeddings generated by "%s" - "%s"', id, methods[id])
			songs               = df['song'].unique().tolist()
			logger.setLevel(logging.ERROR)
			m2v, sm2v           = get_embeddings_opt(id.split('_')[0], params['dataset'], id, songs)
			logger.setLevel(logging.INFO)
			songs               = pd.DataFrame({ 'm2v': [m2v[x] for x in songs], 'sm2v': [sm2v[x] for x in songs]}, index=songs, columns=['m2v','sm2v'])
			i = 0
			metrics = pd.DataFrame(None, index=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'], columns=['Config','Algo', 'Fold', 'Precision', 'Recall', 'F-measure'])
			report_results_opt('Config', 'Algo','Fold','Prec','Rec', 'F1')
			for train, test in kfold:
				helper 	= Helper(train, test, songs, conf['evaluation']['dataset'])
				m2vTN, sm2vTN, csm2vTN, csm2vUK = execute_algo(train.index, test.index, songs, i + 1, params['topN'], params['k'], helper)
				report_opt(id, i+1, m2vTN, sm2vTN, csm2vTN, csm2vUK)
				metrics = m_app_opt(id, metrics, i+1, m2vTN, sm2vTN, csm2vTN, csm2vUK)
				i+=1
			print(metrics[metrics.Algo == 'm2vTN'].mean().values.tolist())
			w.writerow([id, 'm2vTN'] + metrics[metrics.Algo == 'm2vTN'].mean().values.tolist())
			w.writerow([id, 'sm2vTN'] + metrics[metrics.Algo == 'sm2vTN'].mean().values.tolist())
			w.writerow([id, 'csm2vTN'] + metrics[metrics.Algo == 'csm2vTN'].mean().values.tolist())
			w.writerow([id, 'csm2vUK'] + metrics[metrics.Algo == 'csm2vUK'].mean().values.tolist())
		f.close()
	else:
		for method in methods:
			logging.info('Running the recommender systems with embeddings generated by "%s"', method)
			songs               = df['song'].unique().tolist()
			logger.setLevel(logging.ERROR)
			m2v, sm2v           = get_embeddings(method, params['dataset'], songs, conf['embeddings'])
			logger.setLevel(logging.INFO)
			songs               = pd.DataFrame({ 'm2v': [m2v[x] for x in songs], 'sm2v': [sm2v[x] for x in songs]}, index=songs, columns=['m2v','sm2v'])
			i = 0
			metrics = pd.DataFrame(None, index=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'], columns=['Algo', 'Fold', 'Precision', 'Recall', 'F-measure'])
			f = open('tmp/{}/results_{}.csv'.format(params['dataset'], method), 'w+')
			report_results('Algo','Fold','Prec','Rec', 'F1', f)
			for train, test in kfold:
				helper 	= Helper(train, test, songs, conf['evaluation']['dataset'])
				m2vTN, sm2vTN, csm2vTN, csm2vUK = execute_algo(train.index, test.index, songs, i + 1, params['topN'], params['k'], helper)

				report(i+1, m2vTN, sm2vTN, csm2vTN, csm2vUK, f)
				metrics = m_app(metrics, i+1, m2vTN, sm2vTN, csm2vTN, csm2vUK)

				i+=1

			
			


