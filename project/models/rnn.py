from tensorflow.keras.utils     import to_categorical
from tensorflow.keras.layers    import Embedding, LSTM, Dense, GRU, SimpleRNN
from tensorflow.keras.models    import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow   as tf
import numpy        as np
import pandas       as pd
import os.path      as path
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def rnn(DS, MODEL, W_SIZE, EPOCHS, BATCH_SIZE, EMBEDDING_DIM, NUM_UNITS):
    print(DS, MODEL, W_SIZE, EPOCHS, BATCH_SIZE, EMBEDDING_DIM, NUM_UNITS)
    return 1


# W_SIZE          = 5
# BATCH_SIZE      = 64
# EMBEDDING_DIM   = 128
# NUM_UNITS       = 128
# EPOCHS          = 100

# pwd = '../../dataset/{}/'.format('xiami-small')
# if not path.exists(pwd + 'sessions.txt'):
#     df          = pd.read_csv(pwd + 'session_listening_history.csv', sep = ',')
#     sessions    = df.groupby(by='session').agg(list)['song'].values.tolist()
#     f           = open(pwd + 'sessions.txt', mode='w+')
#     for session in sessions:
#         print('\t'.join(session), file=f)

# sessions        = open(pwd + 'sessions.txt').readlines()
# sessions        = [session.replace('\n', '').split('\t') for session in sessions]
# full            = [j for i in sessions for j in i]
# vocab           = sorted(set(full))
# song2ix         = {u:i for i, u in enumerate(vocab)}
# ix_sessions     = np.array([np.array([song2ix[song] for song in session]) for session in sessions])
# sessions_len    = np.array([len(session) for session in sessions])
# ix_sessions     = np.delete(ix_sessions, np.where(sessions_len <= W_SIZE), 0)
# vocab_size      = len(sorted(set(full))) + 1

# sequences = []

# for session in ix_sessions:
#     max_walk_ix = len(session) - W_SIZE
#     ix = 0
#     while ix < max_walk_ix:
#         x = session[ix:ix+ W_SIZE]
#         y = session[ix+ W_SIZE:ix+ W_SIZE + 1]
#         sequences.append([x,y])
#         ix+=1

# sequences = np.array(sequences)
# np.random.shuffle(sequences)
# X, Y    = np.stack(sequences[:,0], axis=0), np.stack(sequences[:,1], axis=0)

# X_train, X_test = X[int(len(X) *.1):], X[:int(len(X) *.1)]
# y_train, y_test = Y[int(len(Y) *.1):], Y[:int(len(Y) *.1)]

# y_train         = to_categorical(y_train, num_classes=vocab_size)
# y_test          = to_categorical(y_test, num_classes=vocab_size)

# def batch(X, y, bs):
#     while True:
#         for ix in range(0, len(X), bs):
#             input  = X[ix:ix+bs]
#             target = y[ix:ix+bs]

#             yield [input, target]

# es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)

# model = Sequential()

# model.add(Embedding(input_dim=vocab_size, output_dim= EMBEDDING_DIM, input_length= W_SIZE))

# model.add(GRU(NUM_UNITS))

# model.add(Dense(vocab_size, activation='softmax'))

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# model.summary()

# model.fit_generator(generator=batch(X_train, y_train, BATCH_SIZE), 
#                     steps_per_epoch=len(X_train) // BATCH_SIZE,
#                     epochs=EPOCHS,
#                     validation_data=batch(X_test, y_test, BATCH_SIZE),
#                     validation_steps=len(X_test) // BATCH_SIZE, 
#                     callbacks=[es])


