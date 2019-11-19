import os
import numpy as np
import math
from sklearn.metrics.pairwise               import cosine_similarity
import warnings

class Helper():
    def __init__(self, train, test, songs, ds):
        self.ds              = ds
        self.train           = train
        self.test            = test
        self.songs           = songs
        self.m2v_songs       = self.songs.m2v.tolist() 
        self.sm2v_songs      = self.songs.sm2v.tolist()
        self.m2v_songs       = np.array(self.m2v_songs, dtype=np.float)
        self.sm2v_songs      = np.array(self.sm2v_songs, dtype=np.float)
        self.songs_ix        = { v:k for k,v in enumerate(songs.index, 0) }
        self.ix_users        = { v:k for k,v in enumerate(np.concatenate([train.index.values, test.index.values]).tolist(), 0)   }
        self.num_users       = len(self.ix_users)
        self.num_songs       = len(songs.index)
        self.ix_pref         = { v:self.u_pref(k) for (k,v) in self.ix_users.items() }
        self.ix_u_songs      = { v:self.unique_songs(k) for (k,v) in self.ix_users.items() }

    def user_sessions(self, user):
        history = self.test.loc[user, 'history']
        return [(s[:len(s)//2], s[len(s)//2:]) for s in history]

    def song_ix(self, song):
        return self.songs_ix[song]

    def ix_user(self, ix):
        return self.ix_users[ix]

    def unique_songs(self, user):
        if user in self.train.index.values:
            history = self.train[self.train.index == user]['history'].values[0]
        if user in self.test.index.values:
            history = self.test[self.test.index == user]['history'].values[0]
        flat_history = [song for session in history for song in session]
        unique_songs = list(set(flat_history))
        return unique_songs

    def u_pref(self,user):
        if user in self.train.index.values:
            history = self.train[self.train.index == user]['history'].values[0]
        if user in self.test.index.values:
            history = self.test[self.test.index == user]['history'].values[0]
            history = [s[:len(s)//2] for s in history]
        flat_history = [song for session in history for song in session]
        flat_history = [self.songs.loc[song, 'm2v'] for song in flat_history]
        mean         = np.mean(flat_history, axis=0)
        return mean

    def c_pref(self, songs):
        flat_vecs       = self.songs.loc[songs, 'sm2v'].tolist()
        return np.mean(np.array(flat_vecs), axis=0)
            
    def get_n_largest(self, cos,n):
        songs = self.songs.index.values
        index = np.argpartition(cos, -n)[-n:]
        return songs[index]

    def uu_matrix(self):
        if os.path.isfile('tmp/{}/matrix_users.npy'.format(self.ds)):
            return np.load('tmp/{}/matrix_users.npy'.format(self.ds))

        matrix_users    = np.zeros((self.num_users, self.num_users))

        for ix in range(self.num_users):
            u_array = np.array([self.ix_pref[i] for i in range(self.num_users)])
            y_array = np.zeros(self.num_users)
            for j in range(self.num_users):
                y_array[j] = math.sqrt(len(self.ix_u_songs[ix]) + len(self.ix_u_songs[j]))
            cos = cosine_similarity(self.ix_pref[ix].reshape(1, -1), u_array)
            val = np.sum([cos, y_array], axis=0) 
            matrix_users[ix] = np.divide(np.ones(val.shape), val)
        np.save('tmp/{}/matrix_users'.format(self.ds), matrix_users)
        return matrix_users

    def us_matrix(self):
        if os.path.isfile('tmp/{}/matrix_user_songs.npy'.format(self.ds)):
            return np.load('tmp/{}/matrix_user_songs.npy'.format(self.ds))
        
        matrix_u_songs  = np.zeros((self.num_users, self.num_songs))
        for u in list(self.ix_u_songs.keys()):
            songs = self.ix_u_songs[u]
            songs_ids = [self.songs_ix[s] for s in songs]
            y_array = np.zeros(self.num_songs)
            y_array[songs_ids] = 1
            matrix_u_songs[u] = y_array
        np.save('tmp/{}/matrix_user_songs'.format(self.ds), matrix_u_songs)
        return matrix_u_songs
