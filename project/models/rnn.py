from os.path                        import exists
from keras.utils                    import to_categorical
from keras.models                   import Model
from keras.layers                   import Embedding, CuDNNLSTM, Dense, CuDNNGRU, SimpleRNN, Input, Bidirectional, Dropout, Concatenate
from keras.models                   import Sequential, load_model
from keras.callbacks                import EarlyStopping, ModelCheckpoint
from keras.utils 					import plot_model
from keras.preprocessing.sequence   import TimeseriesGenerator
import concurrent.futures as fut
import os
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
	# sequences   = df.song.values.tolist()
	# sequences   = np.array(sequences).ravel().astype(str)
	WINDOW 		= W_SIZE * 2
	# x, y = window_seqs(sequences, WINDOW)
	# if not exists(pwd + 'sessions_{}.txt'.format(W_SIZE)):
	# 	f  = open(pwd + 'sessions_{}.txt'.format(W_SIZE), mode='w+')
	# 	for seq, target in zip(x, y):
	# 		print(';'.join(seq) + '\t' + target[0], file=f)
	# 	f.close()
	# sessions        = open(pwd + 'sessions_{}.txt'.format(W_SIZE)).readlines()
	# sessions        = [session.replace('\n', '').split('\t') for session in sessions]
	# sessions        = [data[0].split(';') for data in sessions]
	# targets         = [data[1] for data in sessions]
	# full            = [j for i in (sessions) for j in i]
	# full            = full + targets
	vocab           = sorted(set(df.song.unique().tolist()))
	vocab_size      = len(vocab) +1
	song2ix         = {u:i for i, u in enumerate(vocab, 1)}
	pickle.dump(song2ix, open('song2ix.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
	# sequences       = []
	
	if not exists(pwd + 'song_context_{}.txt'.format(W_SIZE)):
		df['song'] 		= df.song.apply(lambda song: song2ix[song])
		u_playlists 	= df[['user', 'song']].groupby('user').agg(tuple)['song'].values
		u_playlists		= [list(p) for p in u_playlists]
		s_playlists 	= df[['session', 'song']].groupby('session').agg(tuple)['song'].values
		s_playlists		= [list(p) for p in s_playlists]

		nou_playlists 	= len(u_playlists)
		nos_playlists 	= len(s_playlists)

		user_windows        = dict()
		session_windows 	= dict()
		# user_emb 			= dict()
		# session_emb 		= dict()


		for song in vocab:
			user_windows[song2ix[song]] 	= []
			session_windows[song2ix[song]] 	= []

		k4 = 1
		for pl in u_playlists:
			print('[{}/{}] [USER] Playlist'.format(k4, nou_playlists), flush=False, end='\r')
			k4+=1
			ixes 		= range(0, len(pl))
			s_windows 	= [(pl[ix], get_window(pl, ix, W_SIZE)) for ix in ixes]
			for song, window in s_windows:
				user_windows[song].append(window)
		print()
		k4 = 1
		for pl in s_playlists:
			print('[{}/{}] [SESSION] Playlist'.format(k4, nos_playlists), flush=False, end='\r')
			k4+=1
			ixes 		= range(0, len(pl))
			s_windows 	= [(pl[ix], get_window(pl, ix, W_SIZE)) for ix in ixes]
			for song, window in s_windows:
				session_windows[song].append(window)
		print()

		f = open(pwd + 'song_context_{}.txt'.format(W_SIZE), 'w')
		for song in vocab:
		# 	print('[{}/{}] Predicting the embeddings of song {}.'.format(k4, vocab_size, ("%.100s" % song)), flush=False, end='\r')
		# 	k4+=1
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
	embedding_session 		= Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name='Session_Embeddings')(input_session)
	drop_session 			= Dropout(0.2)(embedding_session)
	rec_session 			= CuDNNLSTM(NUM_UNITS, name='Session_LSTM')(drop_session)
	drop_session 			= Dropout(0.2)(rec_session)


	input_user 				= Input(batch_shape=(None, WINDOW))
	embedding_user 			= Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name='User_Embeddings')(input_user)
	drop_user				= Dropout(0.2)(embedding_user)
	rec_user				= CuDNNLSTM(NUM_UNITS, name='User_LSTM')(drop_user)
	drop_user				= Dropout(0.2)(rec_user)


	combination 			= Concatenate()([drop_session, drop_user])
	dense       			= Dense(vocab_size, activation='softmax', name='Densa')(combination)
	model       			= Model(inputs=[input_session, input_user], outputs=dense)

	checkpoint 				= ModelCheckpoint('model_checkpoint.h5', monitor='loss', verbose=1, save_best_only=False, period=1)
	es          			= EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()

	plot_model(model, to_file='model.png')

	model.fit_generator(generator=batch(train, BATCH_SIZE), steps_per_epoch=len(train) // BATCH_SIZE, epochs=EPOCHS, 
						validation_data=batch(test, BATCH_SIZE), validation_steps=len(test) // BATCH_SIZE,  callbacks=[es, checkpoint])

	session_embeddings 	= model.get_layer('Session_Embeddings').get_weights()[0]
	user_embeddings 	= model.get_layer('User_Embeddings').get_weights()[0]


	u_emb = {}
	s_emb = {}

	for song in vocab:
		u_emb[song2ix[song]] = user_embeddings[song2ix[song]]
		s_emb[song2ix[song]] = session_embeddings[song2ix[song]]

	return u_emb, s_emb




	# for seq, target in zip(sessions, targets):
	# 	seq_ix      = [song2ix[song] for song in seq]
	# 	target_ix   = song2ix[target]
	# 	sequences.append([np.array(seq_ix), np.array([target_ix])])

	# sequences   = np.array(sequences)


	# input       = Input(shape=(None, WINDOW))
	# embedding   = Embedding(input_dim=vocab_size, output_dim= EMBEDDING_DIM, input_length=None)(input)

	# if MODEL == 'GRU':
	# 	rec, state  = CuDNNGRU(NUM_UNITS, return_state=True)(embedding)
	# if MODEL == 'RNN':
	# 	rec, state  = SimpleRNN(NUM_UNITS, return_state=True)(embedding)
	# if MODEL == 'LSTM':
		# rec, state_c, state_h 	= CuDNNLSTM(NUM_UNITS, return_state=True)(input)
		# state       				= [state_c, state_h]
		# rec, fw_h, fw_c, bw_h, bw_c = Bidirectional(CuDNNLSTM(NUM_UNITS, return_state=True))(embedding)
		# state 						= [Concatenate()([fw_c, bw_c]), Concatenate()([fw_h, bw_h])]





	# input 			= Input(batch_shape=(None, WINDOW, 1), name='Entrada')
	# rec				= CuDNNLSTM(NUM_UNITS, return_sequences=True, name='LSTM1')(input)
	# drop 			= Dropout(0.1, name='Dropout')(rec)
	# rec_2, s_c, s_h	= CuDNNLSTM(EMBEDDING_DIM, return_state=True, name='LSTM2')(drop)
	# drop2 			= Dropout(0.1, name='Dropout2')(rec_2)
	# state       	= [s_c, s_h]
	# dense       	= Dense(vocab_size, activation='softmax', name='Densa')(drop2)

	# model       	= Model(inputs=input, outputs=dense)
	# inference   	= Model(inputs=input, outputs=state)
	# checkpoint 	= ModelCheckpoint('model_checkpoint.h5', monitor='loss', verbose=1, save_best_only=False, period=1)
	# es          	= EarlyStopping(monitor='acc', mode='max', verbose=1, patience=5)

	# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# model.summary()
	# # inference.summary()
	# model.fit_generator(generator=batch(X_train, y_train, BATCH_SIZE), steps_per_epoch=len(X_train) // BATCH_SIZE,
	# 										epochs=EPOCHS, validation_data=batch(X_test, y_test, BATCH_SIZE),
	# 										validation_steps=len(X_test) // BATCH_SIZE,  callbacks=[es, checkpoint])
	

	# f = open('training_model.yaml', "w")
	# f.write(model.to_yaml())
	# f.close()
	# model.save_weights('training_weights.h5')
	# inference.save_weights('inference_weights.h5')
	
	# model.load_weights('training_model.h5')
	# inference.load_weights('inference_model.h5')
	# print('groupby')
	# u_playlists 	= df[['user', 'song']].groupby('user').agg(tuple)['song'].values
	# u_playlists		= [list(p) for p in u_playlists]
	# s_playlists 	= df[['session', 'song']].groupby('session').agg(tuple)['song'].values
	# s_playlists		= [list(p) for p in s_playlists]

	# nou_playlists 	= len(u_playlists)
	# nos_playlists 	= len(s_playlists)

	# user_windows        = dict()
	# session_windows 	= dict()
	# user_emb 			= dict()
	# session_emb 		= dict()

	# for song in vocab:
	# 	user_windows[song] 		= []
	# 	session_windows[song] 	= []

	# k4 = 1
	# for pl in u_playlists:
	# 	print('[{}/{}] [USER] Playlist'.format(k4, nou_playlists), flush=False, end='\r')
	# 	k4+=1
	# 	ixes 		= range(0, len(pl))
	# 	s_windows 	= [(pl[ix], get_window(pl, ix, W_SIZE)) for ix in ixes]
	# 	for song, window in s_windows:
	# 		user_windows[song].append(window)
	# print()
	# k4 = 1
	# for pl in s_playlists:
	# 	print('[{}/{}] [SESSION] Playlist'.format(k4, nos_playlists), flush=False, end='\r')
	# 	k4+=1
	# 	ixes 		= range(0, len(pl))
	# 	s_windows 	= [(pl[ix], get_window(pl, ix, W_SIZE)) for ix in ixes]
	# 	for song, window in s_windows:
	# 		session_windows[song].append(window)
	# print()
	# k4 = 1
	# for song in vocab:
	# 	print('[{}/{}] Predicting the embeddings of song {}.'.format(k4, vocab_size, ("%.100s" % song)), flush=False, end='\r')
	# 	k4+=1
	# 	u_occurrences = user_windows[song]
	# 	s_occurrences = session_windows[song]

	# 	u_data 	= np.array([[song2ix[song] for song in occ] for occ in u_occurrences])
	# 	s_data 	= np.array([[song2ix[song] for song in occ] for occ in s_occurrences])

	# 	u_bs 	= len(u_occurrences)
	# 	s_bs 	= len(s_occurrences)

	# 	u_pred = np.zeros((1, NUM_UNITS))
	# 	s_pred = np.zeros((1, NUM_UNITS))
	# 	if u_bs > BATCH_SIZE:
	# 		u_splits = np.array_split(u_data, u_bs //	BATCH_SIZE)
	# 		s_splits = np.array_split(s_data, s_bs // 	BATCH_SIZE)

	# 		for split in u_splits:
	# 			pred 	= inference.predict(np.array(split), batch_size=(u_bs // BATCH_SIZE))[0]
	# 			u_pred 	= np.append(u_pred, pred, axis=0)

	# 		for split in s_splits:
	# 			pred 	= inference.predict(np.array(split), batch_size=(s_bs // BATCH_SIZE))[0]
	# 			s_pred 	= np.append(s_pred, pred, axis=0)
		
	# 		u_pred = np.delete(u_pred, (0), axis=0)
	# 		s_pred = np.delete(s_pred, (0), axis=0)

	# 	else: 
	# 		u_pred 	= inference.predict(np.array(u_data), batch_size=u_bs)[0]
	# 		s_pred 	= inference.predict(np.array(s_data), batch_size=s_bs)[0]

	# 	if MODEL == 'LSTM':
	# 		u_pred = np.mean(u_pred, axis=0)
	# 		s_pred = np.mean(s_pred, axis=0)

	# 	user_emb[song] 		= u_pred
	# 	session_emb[song] 	= s_pred
	# print()
	# 



