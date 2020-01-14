from os.path                        import exists
from keras.utils                    import to_categorical
from keras.models                   import Model, load_model
from keras.layers                   import Embedding, CuDNNLSTM, Dense, CuDNNGRU, SimpleRNN, Input, Bidirectional, Concatenate
from keras.models                   import Sequential
from keras.callbacks                import EarlyStopping
from keras.preprocessing.sequence   import TimeseriesGenerator
import concurrent.futures as fut
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
	return x, y

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
		rec, fw_h, fw_c, bw_h, bw_c = Bidirectional(CuDNNLSTM(NUM_UNITS, return_state=True))(embedding)
		state 											= [Concatenate()([fw_c, bw_c]), Concatenate()([fw_h, bw_h])]

	dense       = Dense(vocab_size, activation='softmax')(rec)
	model       = Model(inputs=input, outputs=dense)
	inference   = Model(inputs=input, outputs=state)
	es          = EarlyStopping(monitor='acc', mode='max', verbose=1, patience=3)

	#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	inference.summary()
	#model.fit_generator(generator=batch(X_train, y_train, BATCH_SIZE), steps_per_epoch=len(X_train) // BATCH_SIZE,
											#epochs=EPOCHS, validation_data=batch(X_test, y_test, BATCH_SIZE),
											#validation_steps=len(X_test) // BATCH_SIZE,  callbacks=[es])

	#model.save_weights("training_model.h5")
	#inference.save_weights("inference_model.h5")
	# model = load_model('training_model.yaml')
	# model.load_weights('training_weights.h5')
	inference.load_weights('inference_weights.h5')
	print('groupby')
	u_playlists 	= df.groupby('user').agg(list)['song'].values
	s_playlists 	= df.groupby('session').agg(list)['song'].values
	nou_playlists 	= len(u_playlists)
	nos_playlists 	= len(s_playlists)

	user_windows        = dict()
	session_windows 		= dict()
	user_emb 						= dict()
	session_emb 				= dict()

	for song in vocab:
		user_windows[song] 			= []
		session_windows[song] 	= []

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
	k4 = 1
	for song in vocab:
		print('[{}/{}] Predicting the embeddings of song {}.'.format(k4, vocab_size, ("%.100s" % song)), flush=False, end='\r')
		k4+=1
		u_occurrences = user_windows[song]
		s_occurrences = session_windows[song]

		u_data 	= np.array([[song2ix[song] for song in occ] for occ in u_occurrences])
		s_data 	= np.array([[song2ix[song] for song in occ] for occ in s_occurrences])

		u_bs 	= len(u_occurrences)
		s_bs 	= len(s_occurrences)

		u_pred 	= inference.predict(np.array(u_data), batch_size=u_bs)[0]
		s_pred 	= inference.predict(np.array(s_data), batch_size=s_bs)[0]

		if MODEL == 'LSTM':
			u_pred = np.mean(u_pred, axis=0)
			s_pred = np.mean(s_pred, axis=0)

		user_emb[song] 		= u_pred
		session_emb[song] 	= s_pred
	print()
	return user_emb, session_emb
