import pandas as pd
import numpy as np
from math import floor, inf


df    = pd.read_csv('dataset/xiami-small/session_listening_history.csv', sep=',')
ix    = df.groupby('session')['song'].agg(lambda x: len(list(x)))
df    = df[df.session.isin(ix[ix > 1].index)]
songs = df.song.unique()

fc = open('dataset/xiami-small/contextual_seqs.txt')
fu = open('dataset/xiami-small/listening_seqs.txt')


cseqs = [line.replace('\n', '').split(' ') for line in fc.readlines()]
cseqs = [np.array(l) for l in cseqs if len(l) > 1]
useqs = [line.replace('\n', '').split(' ') for line in fu.readlines() ]
useqs = [np.array(l) for l in useqs if len(l) > 1]
fc.close()
fu.close()


def get_window(song, seqs, w_size):
    w = w_size // 2
    to_r = []
    for seq in seqs:
        ixs = np.where(song == seq)[0]
        seql = seq.tolist()
        for ix in ixs:
            if ix - w < 0:
                ab = abs(ix - w)
                b4 = ['-'] * ab + (seql[0:ix])
            else:
                b4 = seql[ix - w:ix]
            if ix + w > len(seq) - 1:
                ab = abs(ix + w) - (len(seq) - 1)
                af = seql[ix + 1: len(seq)] + (['-'] * ab )
            else:
                af = seql[ix + 1: ix + w + 1]
            to_r.append(np.array(b4 + [seql[ix]] + af))
    return np.array(to_r)

f1 = open('c_seqs.csv', 'w+')
f2 = open('u_seqs.csv', 'w+')

for song in songs:
    cs_seqs = get_window(song, cseqs, 5)
    for seq in cs_seqs:
        print(song + '\t'+ '[{}]'.format( ','.join(seq)), file=f1)

    cu_seqs = get_window(song, useqs, 5)
    for seq in cu_seqs:
        print(song + '\t'+ '[{}]'.format( ','.join(seq)), file=f2)



