from os.path                        import exists
from keras.utils                    import to_categorical
from keras.models                   import Model
from keras.layers                   import Embedding, CuDNNLSTM, Dense, CuDNNGRU, SimpleRNN, Input, TimeDistributed
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

def window_seqs(sequence, w_size):
    ix = 0
    max_ix = (len(sequence) - 1) - w_size
    x = []
    y = []
    while ix < max_ix:
        x.append(sequence[ix:ix+w_size])
        y.append([sequence[ix+w_size]])
        ix+=1
    return np.vstack(x), np.vstack(y)

def rnn(df, DS, MODEL, W_SIZE, EPOCHS, BATCH_SIZE, EMBEDDING_DIM, NUM_UNITS):
    pwd = 'dataset/{}/'.format(DS)

    sequences = df.song.values.tolist()
    sequences = np.array(sequences).ravel().astype(str)
    
    x, y = window_seqs(sequences, W_SIZE)
    if not exists(pwd + 'sessions_{}.txt'.format(W_SIZE)):
        f           = open(pwd + 'sessions_{}.txt'.format(W_SIZE), mode='w+')
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

    input       = Input(shape=(W_SIZE,))
    embedding   = Embedding(input_dim=vocab_size, output_dim= EMBEDDING_DIM, input_length= W_SIZE)(input)
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
    es          = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=5)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit_generator(generator=batch(X_train, y_train, BATCH_SIZE), steps_per_epoch=len(X_train) // BATCH_SIZE,
                        epochs=EPOCHS, validation_data=batch(X_test, y_test, BATCH_SIZE),
                        validation_steps=len(X_test) // BATCH_SIZE,  callbacks=[es])

    model.save_weights("best.h5")
    # User Embeddings

    all_users_playlists = df.groupby('user').agg(list)['song'].values
    song_windows = {}
    for song in vocab:
        if song not in song_windows:
            song_windows[song] = []

        for playlist in all_users_playlists:
            ixes = [i for i, x in enumerate(playlist) if (x == song)]
            for ix in ixes:
                if ix-W_SIZE > 0:
                    song_windows[song].append(playlist[ix-W_SIZE:ix])
                # if ix+W_SIZE < len(playlist):
                #     song_windows[song].append(playlist[ix:ix+W_SIZE])

    user_emb = {}

    for k, occurrences in song_windows.items():
        bs = len(occurrences)
        data = np.array([[song2ix[song] for song in occ] for occ in occurrences])
        # O vetor de embeddings que serã inferido será uma repetição da mesma música n vezes
        if bs == 0:
            data = np.array([[song2ix[song]] * W_SIZE])
            bs = 1
        state = inference.predict(np.array(data), batch_size=bs)
        emb = np.mean(np.array(state), axis=0)
        user_emb[k] = emb

    # # Session Embeddings

    all_sessions_playlists = df.groupby('session').agg(list)['song'].values
    song_windows = {}
    for song in vocab:
        if song not in song_windows:
            song_windows[song] = []
        for playlist in all_sessions_playlists:
            ixes = [i for i, x in enumerate(playlist) if (x == song)]
            for ix in ixes:
                if ix-W_SIZE > 0:
                    song_windows[song].append(playlist[ix-W_SIZE:ix])
                if ix+W_SIZE < len(playlist):
                    song_windows[song].append(playlist[ix:ix+W_SIZE])
    session_emb = {}

    for k, occurrences in song_windows.items():
        bs = len(occurrences)
        data = np.array([[song2ix[song] for song in occ] for occ in occurrences])
        data = np.array(data)
        # O vetor de embeddings que será inferido será uma repetição da mesma música n vezes
        if bs == 0:
            data = np.array([[song2ix[song]] * W_SIZE])
            bs = 1
        state = inference.predict(data, batch_size=bs)
        emb = np.mean(np.array(state), axis=0)
        session_emb[k] = emb
    return user_emb, session_emb



