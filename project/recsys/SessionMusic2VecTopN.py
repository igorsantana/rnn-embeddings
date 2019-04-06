import time
import numpy            as np
import multiprocessing  as mp
from gensim.similarities.index  import AnnoyIndexer

class SessionMusic2VecTopN():
    def __init__(self, train, m2v, topN):
        self.train  = train
        self.m2v    = m2v
        self.n      = topN
   
    def top_n(self, p):
        sim     = self.m2v.similar_by_vector(p, self.n)
        sim     = sorted(sim, key= lambda x: x[1], reverse=True)
        return [song[0] for song in sim]