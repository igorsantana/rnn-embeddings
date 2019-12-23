from os.path                        import exists
from keras.utils                    import to_categorical
from keras.models                   import Model
from keras.layers                   import Embedding, CuDNNLSTM, Dense, CuDNNGRU, SimpleRNN, Input, Bidirectional
from keras.models                   import Sequential
from keras.callbacks                import EarlyStopping
from keras.preprocessing.sequence   import TimeseriesGenerator
import os
import time
import numpy        as np
import pickle       as pk
import pandas       as pd
import tensorflow   as tf


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
		b4 = (abs(ix - window) * [el]) + playlist[0:ix]
	else:
		b4 = playlist[ix - window:ix]
	# If the problem is in the after clause, append the song until it mets the window size.

	if (ix + window + 1) > len(playlist):
		num		=	(ix + window + 1) - len(playlist)
		after 	= 	playlist[ix + 1:ix + window + 1] + (num * [el])
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
	return x,y

def rnn(df, DS, MODEL, W_SIZE, EPOCHS, BATCH_SIZE, EMBEDDING_DIM, NUM_UNITS, BIDIRECTIONAL):
	pwd 		= 'dataset/{}/'.format(DS)
	sequences   = df.song.values.tolist()
	sequences   = np.array(sequences).ravel().astype(str)
	WINDOW 		= W_SIZE * 2



	x, y = window_seqs(sequences, WINDOW)
	if not exists(pwd + 'sessions_{}.txt'.format(W_SIZE)):
		f  = open(pwd + 'sessions_{}.txt'.format(W_SIZE), mode='w+')
		for seq, target in zip(x, y):
			print(';'.join(seq) + '\t' + target[0], file=f)
		f.close()

	sessions        = open(pwd + 'sessions_{}.txt'.format(W_SIZE)).readlines()
	sessions        = [session.replace('\n', '').split('\t') for session in sessions]
	sessions        = [data[0].split(';') for data in sessions]
	targets         = [data[1] for data in sessions]
	full            = [j for i in (sessions) for j in i]
	full            = full + targets
	vocab           = sorted(set(full))
	vocab_size      = len(vocab)
	song2ix         = {u:i for i, u in enumerate(vocab)}
	sequences       = []

	for seq, target in zip(sessions, targets):
		seq_ix      = [song2ix[song] for song in seq]
		target_ix   = song2ix[target]
		sequences.append([np.array(seq_ix), np.array([target_ix])])

	sequences   = np.array(sequences)
	np.random.shuffle(sequences)
	X, Y        = np.stack(sequences[:,0], axis=0), np.stack(sequences[:,1], axis=0)

	X_train, X_test = X[int(len(X) *.1):], X[:int(len(X) *.1)]
	y_train, y_test = Y[int(len(Y) *.1):], Y[:int(len(Y) *.1)]

	def batch(X, y, bs):
		while True:
			for ix in range(0, len(X), bs):
				input  = X[ix:ix+bs]
				target = y[ix:ix+bs]
				yield input, to_categorical(target, num_classes=vocab_size)

	input       = Input(shape=(WINDOW,))
	embedding   = Embedding(input_dim=vocab_size, output_dim= EMBEDDING_DIM, input_length= WINDOW)(input)

	if MODEL == 'GRU':
		rec, state  = CuDNNGRU(NUM_UNITS, return_state=True)(embedding)
	if MODEL == 'RNN':
		rec, state  = SimpleRNN(NUM_UNITS, return_state=True)(embedding)
	if MODEL == 'LSTM':
		rec, state_c, state_h  = CuDNNLSTM(NUM_UNITS, return_state=True)(embedding)
		state       = [state_c, state_h]

	dense       = Dense(vocab_size, activation='softmax')(rec)
	model       = Model(inputs=input, outputs=dense)
	inference   = Model(inputs=input, outputs=state)
	es          = EarlyStopping(monitor='acc', mode='max', verbose=1, patience=3)

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	#model.fit_generator(generator=batch(X_train, y_train, BATCH_SIZE), steps_per_epoch=len(X_train) // BATCH_SIZE,
	#										epochs=EPOCHS, validation_data=batch(X_test, y_test, BATCH_SIZE),
	#										validation_steps=len(X_test) // BATCH_SIZE,  callbacks=[es])

	#model.save_weights("training_model.h5")
	#inference.save_weights("inference_model.h5")
	model.load_weights('training_model.h5')
	inference.load_weights('inference_model.h5')
	print('groupby')
	all_users_playlists 		= df.groupby('user').agg(list)['song'].values
	all_sessions_playlists 	= df.groupby('session').agg(list)['song'].values

	song_windows        = {}
	print('user song stage 0')
	xx = 0
	for song in vocab:
		print('[{}/{}] Song {} is having its windows processed'.format(xx, vocab_size, song), flush=True, end='\r')
		if song not in song_windows:
			song_windows[song] = []

		indexes = [[ix for ix, x in enumerate(playlist) if (x == song)] for playlist in all_users_playlists]
		
		for index, list_indexes in enumerate(indexes):
			if len(list_indexes) > 0:
				playlist = all_users_playlists[index]
				for ix in list_indexes:
					window = get_window(playlist, ix, W_SIZE)
					song_windows[song].append(window)
		xx+=1

	user_emb = dict()
	print('user song stage 1')
	xx = 0
	for k, occurrences in song_windows.items():
		print('[{}/{}] Song {} is being predicted'.format(xx, vocab_size, k), flush=True, end='\r')
		bs = len(occurrences)
		data = np.array([[song2ix[song] for song in occ] for occ in occurrences])
		pred = inference.predict(np.array(data), batch_size=bs)[0]
		if MODEL == 'LSTM':
			pred = np.mean(pred, axis=0)
		user_emb[k] = pred
		xx+=1

	song_windows        = {}
	print('session song stage 0')
	xx = 0
	for song in vocab:
		print('[{}/{}] Song {} is having its windows processed'.format(xx, vocab_size, song), flush=True, end='\r')
		if song not in song_windows:
			song_windows[song] = []

		indexes = [[ix for ix, x in enumerate(playlist) if (x == song)] for playlist in all_sessions_playlists]
		
		for index, list_indexes in enumerate(indexes):
			if len(list_indexes) > 0:
				playlist = all_sessions_playlists[index]
				for ix in list_indexes:
					window = get_window(playlist, ix, W_SIZE)
					song_windows[song].append(window)
		xx+=1

	session_emb = dict()
	print('session song stage 1')
	xx=0
	for k, occurrences in song_windows.items():
		print('[{}/{}] Song {} is being predicted'.format(xx, vocab_size, k), flush=True, end='\r')
		bs = len(occurrences)
		data = np.array([[song2ix[song] for song in occ] for occ in occurrences])
		
		pred = inference.predict(np.array(data), batch_size=bs)[0]

		if MODEL == 'LSTM':
			pred = np.mean(pred, axis=0)
		session_emb[k] = pred
		xx+=1
		
	return user_emb, session_emb




