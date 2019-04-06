import os
import csv
import pandas as pd
import numpy as np
import multiprocessing      as mp
from datetime import datetime

format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x))


def generate_files(df, fold, i, ds):
	printlog('Starting to generate fold {}'.format(i))
	u_train = df[df.user.isin(fold[0].tolist())]
	u_train.to_csv('tmp/cv/{}/train_{}.csv'.format(ds, i), header=True, index=False)
	u_test  = df[df.user.isin(fold[1].tolist())]
	k = 0
	for user in np.unique(u_test['user'].values):
		data = u_test[u_test.user == user]
		to_train    = data.iloc[0:(len(data.index) // 2) ]
		to_train.to_csv('tmp/cv/{}/train_{}.csv'.format(ds, i), header=False, index=False, mode='a')
		to_test     = data.iloc[(len(data.index) // 2):(len(data.index)) ]
		if k == 0:
			to_test.to_csv('tmp/cv/{}/test_{}.csv'.format(ds, i), header=True, index=False, mode='w')
		to_test.to_csv('tmp/cv/{}/test_{}.csv'.format(ds, i), header=False, index=False, mode='a')
		k+=1
	printlog('Finished to generate fold {}'.format(i))

def cross_validation(dataset, k):
	if sum([file.endswith('.csv') for file in os.listdir('tmp/cv/{}/'.format(dataset))]) > 0:
		printlog('Cross-validation already done, exiting this phase.')
		return
	printlog('Reading the dataset to split into {} folds.'.format(k))
	df		= pd.read_csv('dataset/{}/session_listening_history.csv'.format(dataset), sep = ',')
	printlog('Starting to split users into folds.')
	users       = np.unique(df['user'].values)
	np.random.shuffle(users)

	users       = np.array_split(users, int(k))
	user_folds  = [(np.concatenate(np.delete(users, i, axis=0)), users[i]) for i in range(int(k))]
	printlog('Users were splitted into folds, now saving them at tmp/cv/{}.'.format(dataset))
	if not os.path.isdir('tmp/cv/{}'.format(dataset)): os.mkdir('tmp/cv/{}'.format(dataset))
	processes = []
	for i in range(int(k)):
		p = mp.Process(target=generate_files, args=(df, user_folds[i], i, dataset))
		processes.append(p)
	for p in processes:	p.start()
	for p in processes: p.join()
	return