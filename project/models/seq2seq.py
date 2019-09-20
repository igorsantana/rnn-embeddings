import numpy as np
import pandas as pd
import logging
from project.data.preprocess        import window_sequences, gen_seq_files
from os.path                        import exists
from keras.models                   import Model
from keras.callbacks                import EarlyStopping
from keras.layers                   import Dense, CuDNNLSTM, CuDNNGRU, Embedding, Input, SimpleRNN

def read_input_targets(path, win_size, t):
    if t == 'session':
        f    = open(path + 'contextual_seqs.txt')
        s_i = []
        for line in f: s_i.append(line.rstrip('\n'))
        f.close()
        s_i      = window_sequences(s_i, win_size)
        s_t      = ['START_ ' + session + ' _END' for session in s_i]
        return s_i, s_t
    if t == 'listening':
        f    = open(path + 'listening_seqs.txt')
        s_i = []
        for line in f: s_i.append(line.rstrip('\n'))
        f.close()
        s_i = window_sequences(s_i, win_size)
        s_t = ['START_ ' + session + ' _END' for session in s_i]
        return s_i, s_t

def get_unique_songs(s_i, s_t):
    all_i   = set()
    all_t   = set()
    for songs in s_i:
        for song in songs.split():
            if song not in all_i:
                all_i.add(song)
    for songs in s_t:
        for song in songs.split():
            if song not in all_t:
                all_t.add(song)
    return sorted(list(all_i)), sorted(list(all_t))

def get_max_length(s_i, s_t):
    max_i = np.max([len(session.split()) for session in s_i])
    max_t = np.max([len(session.split()) for session in s_t])
    return max_i, max_t

def get_dicts(i_songs, t_songs):
    song_ix_i = dict([(song, i+1) for i, song in enumerate(i_songs)])
    song_ix_t = dict([(word, i+1) for i, word in enumerate(t_songs)])
    ix_song_i = dict((i, song) for song, i in song_ix_i.items())
    ix_song_t = dict((i, song) for song, i in song_ix_t.items())
    return song_ix_i, song_ix_t, ix_song_i, ix_song_t

def __run_s2s(sessions_i, sessions_t, num_songs, song_ix, max_l, NUM_DIM=128, BATCH_SIZE= 128, EPOCHS=50, MODEL='RNN', WINDOW_SIZE=5):
    X, y                                 = sessions_i, sessions_t
    num_encoder_songs, num_decoder_songs = num_songs
    song_ix_i, song_ix_t                 = song_ix
    max_length_i, max_length_t           = max_l
    
    def generate_batch(X, y, batch_size= 128):
        while True:
            for j in range(0, len(X), batch_size): 
                encoder_input_data = np.zeros((batch_size, max_length_i), dtype='float32')
                decoder_input_data = np.zeros((batch_size, max_length_t), dtype='float32')
                decoder_target_data = np.zeros((batch_size, max_length_t, num_decoder_songs), dtype='float32')
                for i, (input_sequence, target_sequence) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                    for t, word in enumerate(input_sequence.split()):
                        encoder_input_data[i, t] = song_ix_i[word] if word != '-' else 0
                    for t, word in enumerate(target_sequence.split()):
                        if t < len(target_sequence.split()) - 1:
                            decoder_input_data[i, t] = song_ix_t[word] if word != '-' else 0
                        if t > 0:
                            decoder_target_data[i, t - 1, song_ix_t[word] if word != '-' else 0] = 1
                yield([encoder_input_data, decoder_input_data], decoder_target_data)

    np.random.shuffle(X)
    np.random.shuffle(y)

    X_train, X_test = X[int(len(X) *.1):], X[:int(len(X) *.1)]
    y_train, y_test = y[int(len(y) *.1):], y[:int(len(y) *.1)]

    TRAIN_SAMPLES   = len(X_train)
    VAL_SAMPLES     = len(X_test)

    ENCODER_INPUT       = Input(shape=(None,))
    ENCODER_EMBEDDING   = Embedding(num_encoder_songs, NUM_DIM)(ENCODER_INPUT)
    if MODEL == 'LSTM':
        ENCODER_NN          = CuDNNLSTM(NUM_DIM, return_state=True)
        _, state_h, state_c = ENCODER_NN(ENCODER_EMBEDDING)
        ENCODER_STATE       = [state_h, state_c]
    if MODEL == 'GRU':
        ENCODER_NN          = CuDNNGRU(NUM_DIM, return_state=True)
        _, ENCODER_STATE    = ENCODER_NN(ENCODER_EMBEDDING)
    if MODEL == 'RNN':
        ENCODER_NN          = SimpleRNN(NUM_DIM, return_state=True)
        _, ENCODER_STATE    = ENCODER_NN(ENCODER_EMBEDDING)

    DECODER_INPUT       = Input(shape=(None,))
    DECODER_EMBEDDING   = Embedding(num_decoder_songs, NUM_DIM)(DECODER_INPUT)
    if MODEL == 'LSTM':
        DECODER_NN          = CuDNNLSTM(NUM_DIM, return_sequences=True, return_state=True)
        DECODER_OUTPUT,_,_  = DECODER_NN(DECODER_EMBEDDING, initial_state=ENCODER_STATE)
    if MODEL == 'GRU':
        DECODER_NN          = CuDNNGRU(NUM_DIM, return_sequences=True, return_state=True)
        DECODER_OUTPUT,_    = DECODER_NN(DECODER_EMBEDDING, initial_state=ENCODER_STATE)
    if MODEL == 'RNN':
        DECODER_NN          = SimpleRNN(NUM_DIM, return_sequences=True, return_state=True)
        DECODER_OUTPUT,_    = DECODER_NN(DECODER_EMBEDDING, initial_state=ENCODER_STATE)
    DENSE_DECODER       = Dense(num_decoder_songs, activation='softmax')
    DECODER_OUTPUT      = DENSE_DECODER(DECODER_OUTPUT)

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=2)

    model               = Model([ENCODER_INPUT, DECODER_INPUT], DECODER_OUTPUT)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit_generator(generator= generate_batch(X_train, y_train, batch_size= BATCH_SIZE),
                        steps_per_epoch= TRAIN_SAMPLES // BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data= generate_batch(X_test, y_test, batch_size= BATCH_SIZE),
                        validation_steps= VAL_SAMPLES // BATCH_SIZE, callbacks=[es])
    return model

def start(df, conf, id):
    s2s = conf['models']['seq2seq']
    ds  = conf['evaluation']['dataset']
    if not exists('dataset/{}/listening_seqs.txt'.format(ds)):
        logging.info('Files %s and %s are going to be at "%s"', 'listening_seqs.txt', 'contextual_seqs.txt', 'dataset/{}/'.format(ds))
        gen_seq_files(df, 'dataset/{}/'.format(ds))
    del df
    logging.info('Reading the input data from the RNN')
    sessions_i,     sessions_t                  = read_input_targets('dataset/{}/'.format(ds), s2s['window_size'], 'session')
    listening_i,    listening_t                 = read_input_targets('dataset/{}/'.format(ds), s2s['window_size'], 'listening')
    input_songs,    target_songs                = get_unique_songs(sessions_i, sessions_t)
    max_length_i,   max_length_t                = get_max_length(sessions_i, sessions_t)
    num_encoder_songs, num_decoder_songs        = len(input_songs) + 1, len(target_songs) + 1
    song_ix_i, song_ix_t, ix_song_i, ix_song_t  = get_dicts(input_songs, target_songs)

    model = __run_s2s(listening_i, listening_t, (num_encoder_songs, num_decoder_songs), (song_ix_i, song_ix_t), 
          (max_length_i, max_length_t), NUM_DIM=s2s['vector_dim'], BATCH_SIZE=s2s['batch_size'], EPOCHS=s2s['epochs'],
           MODEL=s2s['model'], WINDOW_SIZE=s2s['window_size'])

    # embeddings = model.layers[2].get_weights()[0]

    # f = open('tmp/{}/models/emb1.csv'.format(ds), 'w')
    # for song in input_songs:
    #     if song != '-': print('{};{}'.format(song, embeddings[song_ix_i[song]].tolist()), file=f)
    # f.close()

    embeddings2 = model.layers[3].get_weights()[0]
    f = open('tmp/{}/models/{}.csv'.format(ds, 'embeddings_seq2seq' if not id else 'seq2seq_' + str(id)), 'w')
    for song in input_songs:
        if song != '-': print('{};{}'.format(song, embeddings2[song_ix_i[song]].tolist()), file=f)
    f.close()

    model = __run_s2s(sessions_i, sessions_t, (num_encoder_songs, num_decoder_songs), (song_ix_i, song_ix_t), 
          (max_length_i, max_length_t), NUM_DIM=s2s['vector_dim'], BATCH_SIZE=s2s['batch_size'], EPOCHS=s2s['epochs'],
           MODEL=s2s['model'], WINDOW_SIZE=s2s['window_size'])

    # embeddings = model.layers[2].get_weights()[0]
    
    # f = open('tmp/{}/models/semb1.csv'.format(ds), 'w')
    # for song in input_songs:
    #     if song != '-': print('{};{}'.format(song, embeddings[song_ix_i[song]].tolist()), file=f)
    # f.close()

    embeddings2 = model.layers[3].get_weights()[0]
    f = open('tmp/{}/models/{}.csv'.format(ds, 'sembeddings_seq2seq' if not id else 'sseq2seq_' + str(id)), 'w')
    for song in input_songs:
        if song != '-': print('{};{}'.format(song, embeddings2[song_ix_i[song]].tolist()), file=f)
    f.close()