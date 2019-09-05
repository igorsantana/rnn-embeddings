import os
import numpy as np
import math
from sklearn.metrics.pairwise               import cosine_similarity

class Matrixes():
    def __init__(self, users, songs, ds):
        self.ds              = ds
        self.users           = users
        self.songs           = songs
        self.num_users       = len(users.index)
        self.num_songs       = len(songs.index)
        self.ix_users        = { v:k for k,v in enumerate(users.index) }
        self.songs_ix        = { v:k for k,v in enumerate(songs.index) }
        self.ix_pref         = { v:self.u_pref(k)[0] for (k,v) in self.ix_users.items() }
        self.ix_u_songs      = { v:self.u_pref(k)[1] for (k,v) in self.ix_users.items() }
        self.m2v_songs       = self.songs.m2v.tolist()
        self.m2v_songs       = np.array(self.m2v_songs, dtype=np.float)
        self.sm2v_songs      = self.songs.sm2v.tolist()
        self.sm2v_songs      = np.array(self.sm2v_songs, dtype=np.float)

    def song_ix(self, song):
        return self.songs_ix[song]
    def ix_user(self, ix):
        return self.ix_users[ix]

    def u_pref(self,user):
        history      = self.users[self.users.index == user]['history'].values.tolist()[0]
        flat_history = [song for session in history for song in session[:len(session)//2]]
        unique_songs = list(set(flat_history))
        flat_history = [self.songs.loc[song, 'm2v'] for song in flat_history]
        return np.mean(flat_history, axis=0), unique_songs

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

        for u in range(self.num_users):
            songs_ids = [self.songs_ix[s] for s in self.ix_u_songs[u]]
            y_array = np.zeros(self.num_songs)
            for s in songs_ids:
                y_array[s] = 1
            matrix_u_songs[u] = y_array
        np.save('tmp/{}/matrix_user_songs'.format(self.ds), matrix_u_songs)
        return matrix_u_songs
