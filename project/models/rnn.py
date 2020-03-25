from os.path                        import exists
from keras.utils                    import to_categorical
from keras.models                   import Model
from keras.layers                   import Embedding, LSTM, Dense, CuDNNGRU, LSTM, Input, Bidirectional, Dropout, Concatenate, Bidirectional
from keras.models                   import Sequential, load_model
from keras.callbacks                import EarlyStopping, ModelCheckpoint
from keras.utils 					import plot_model
from keras.utils 					import multi_gpu_model
from keras.preprocessing.sequence   import TimeseriesGenerator
import concurrent.futures as fut
import os
import gc
import keras
import pickle
import time
import numpy        as np
import pickle       as pk
import pandas       as pd
import tensorflow   as tf
import matplotlib.pyplot as plt
from math 			import floor


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def get_window(playlist, ix, window):
	el = playlist[ix]

	# This is the perfect case:
	if (ix - window >= 0) and (ix + window + 1) < len(playlist):
		window = playlist[ix - window:ix] + playlist[ix + 1:(ix + 1) + window]
		return window

	# Not running into the perfect case, will turn into the damage reduction clause:
	b4      = []
	after   = []
	# If the problem is in the before clause, prepend the song until it mets the window size.
	if (ix - window < 0):
		b4 = (abs(ix - window) * ['0']) + playlist[0:ix]
	else:
		b4 = playlist[ix - window:ix]
	# If the problem is in the after clause, append the song until it mets the window size.

	if (ix + window + 1) > len(playlist):
		num		=	(ix + window + 1) - len(playlist)
		after 	= 	playlist[ix + 1:ix + window + 1] + (num * ['0'])
	else:
		after 	= 	playlist[ix + 1:(ix + 1) + window]

	return b4 + after


def window_seqs(sequence, w_size):
	ix = 0
	max_ix = (len(sequence) - 1) - w_size
	x = []
	y = []
	while ix < max_ix:
		x.append(sequence[ix:ix+w_size])
		y.append([sequence[ix+w_size]])
		ix+=1
	return x, y

def rnn(df, DS, MODEL, W_SIZE, EPOCHS, BATCH_SIZE, EMBEDDING_DIM, NUM_UNITS, BIDIRECTIONAL):
	pwd 		= 'dataset/{}/'.format(DS)
	WINDOW 		= W_SIZE * 2

	vocab           = sorted(set(df.song.unique().tolist()))
	vocab_size      = len(vocab) +1
	song2ix         = {u:i for i, u in enumerate(vocab, 1)}
	pickle.dump(song2ix, open('{}_song2ix.pickle'.format(DS), 'wb'), pickle.HIGHEST_PROTOCOL)
	
	
	if not exists(pwd + 'song_context_{}.txt'.format(W_SIZE)):
		df['song'] 		= df.song.apply(lambda song: song2ix[song])
		u_playlists 	= df[['user', 'song']].groupby('user').agg(tuple)['song'].values
		u_playlists		= [list(p) for p in u_playlists]
		s_playlists 	= df[['session', 'song']].groupby('session').agg(tuple)['song'].values
		s_playlists		= [list(p) for p in s_playlists]

		nou_playlists 	= len(u_playlists)
		nos_playlists 	= len(s_playlists)

		user_windows       	= dict()
		session_windows 	= dict()


		for song in vocab:
			user_windows[song2ix[song]] 	= []
			session_windows[song2ix[song]] 	= []

		k4 = 1
		for pl in u_playlists:
			print('[{}/{}] [USER] Playlist'.format(k4, nou_playlists), flush=False, end='\r')
			k4+=1
			ixes 			= range(0, len(pl))
			s_windows = [(pl[ix], get_window(pl, ix, W_SIZE)) for ix in ixes]
			for song, window in s_windows:
				user_windows[song].append(window)
		print()
		k4 = 1
		for pl in s_playlists:
			print('[{}/{}] [SESSION] Playlist'.format(k4, nos_playlists), flush=False, end='\r')
			k4+=1
			ixes 			= range(0, len(pl))
			s_windows = [(pl[ix], get_window(pl, ix, W_SIZE)) for ix in ixes]
			for song, window in s_windows:
				session_windows[song].append(window)
		print()

		f = open(pwd + 'song_context_{}.txt'.format(W_SIZE), 'w')
		for song in vocab:
			u_occurrences = user_windows[song2ix[song]]
			s_occurrences = session_windows[song2ix[song]]
			for u_o, s_o in zip(u_occurrences, s_occurrences):
				print('{}\t{}\t{}'.format(','.join([str(i) for i in u_o]), ','.join([str(i) for i in s_o]), str(song2ix[song])), file=f)
		f.close()

	f  = open(pwd + 'song_context_{}.txt'.format(W_SIZE), mode='r')

	data = []
	for line in f:
		line = line.replace('\n', '')
		input_user, input_session, target = line.split('\t')
		line = [np.array([int(x) for x in input_user.split(',')]), np.array([int(x) for x in input_session.split(',')]), int(target)]
		data.append(line)
	
	data = np.vstack(data)

	np.random.shuffle(data)

	def batch(data, bs):
		while True:
			for ix in range(0, len(data), bs):
				u_input = data[ix:ix+bs,0]
				s_input = data[ix:ix+bs,1]
				target 	= data[ix:ix+bs,2]
				yield [np.vstack(u_input), np.vstack(s_input)], to_categorical(target, num_classes=vocab_size)


	train, test = data[int(len(data) *.20):], data[:int(len(data) *.20)]

	input_session 			= Input(batch_shape=(None, WINDOW))
	embedding_session 		= Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name='Session_Embeddings', mask_zero=True)(input_session)
	drop_session 			= Dropout(0.2)(embedding_session)
	rec_session 			= LSTM(NUM_UNITS, name='Session_LSTM')(drop_session)
	drop_session 			= Dropout(0.2)(rec_session)

	input_user 				= Input(batch_shape=(None, WINDOW))
	embedding_user 			= Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name='User_Embeddings', mask_zero=True)(input_user)
	drop_user				= Dropout(0.2)(embedding_user)
	rec_user				= LSTM(NUM_UNITS, name='User_LSTM')(drop_user)
	drop_user				= Dropout(0.2)(rec_user)
	combination 			= Concatenate()([drop_session, drop_user])
	dense       			= Dense(vocab_size, activation='softmax', name='Densa')(combination)
	model       			= Model(inputs=[input_session, input_user], outputs=dense)
	checkpoint 				= ModelCheckpoint('{}_model_checkpoint.h5'.format(DS), monitor='loss', verbose=0, save_best_only=False, period=1)
	es          			= EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)

	# model 					= multi_gpu_model(model, 2)

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	
	plot_model(model, to_file='model.png')
	
	
	if exists('{}_model_checkpoint.h5'.format(DS)):
		model = load_model('{}_model_checkpoint.h5'.format(DS))

	model.fit_generator(generator=batch(train, BATCH_SIZE), steps_per_epoch=len(train) // BATCH_SIZE, epochs=EPOCHS, 
						validation_data=batch(test, BATCH_SIZE), validation_steps=len(test) // BATCH_SIZE,  callbacks=[es, checkpoint])

	session_embeddings 		= model.get_layer('Session_Embeddings').get_weights()[0]
	user_embeddings 		= model.get_layer('User_Embeddings').get_weights()[0]

	u_emb = {}
	s_emb = {}

	for song in vocab:
		u_emb[song] = user_embeddings[song2ix[song]]
		s_emb[song] = session_embeddings[song2ix[song]]

	del model
	gc.collect()
	return u_emb, s_emb
