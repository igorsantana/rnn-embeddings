import yaml
import numba
import numpy as np
import multiprocessing                      as mp
import project.data.preprocess              as pp
import project.models.embeddings.models     as emb
import project.evaluation.run               as r
from datetime import datetime




format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x))

if __name__ == '__main__':
    printlog('Starting the process, reading the configuration file.')
    conf = yaml.safe_load(open('config.yml'))

#     # # mp.set_start_method('spawn')

    pp.preprocess(conf['evaluation']['dataset'], conf['session']['interval'])

    emb.model_runner(conf['evaluation']['dataset'], conf['models']['music2vec'])

    r.execute_cv(conf['evaluation'], conf['logfile'])

    

    # u = np.random.rand(100)
    # print(u)
    # print(u.dtype)
    # M = np.random.rand(100000, 100)
    # print(M)
    # print(M.dtype)

    # print(fast_cosine_matrix(u, M))