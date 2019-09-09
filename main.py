import yaml
import pandas as pd
import numpy as np
import multiprocessing                      as mp
import project.data.preprocess              as pp
import project.models.embeddings.models     as emb
import project.evaluation.run               as r
from datetime import datetime
import re



format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x))

if __name__ == '__main__':
    printlog('Starting the process, reading the configuration file.')
    conf = yaml.safe_load(open('config.yml'))

    pp.preprocess(conf['evaluation']['dataset'], conf['session']['interval'])

    emb.model_runner(conf['evaluation']['dataset'], conf['models'])

    r.execute_cv(conf['evaluation'], conf['logfile'], conf['models']['is_doc'])

    vals = {}
    cache = []
    i = 0
    for line in reversed(list(open(conf['logfile']))):
        l = line.rstrip()
        if 'Algo' not in l: 
            cache.append(l)
        else:
            vals[i] = cache
            cache = []
            i+=1
    regex = re.compile(".*?\[(.*?)\]")
    for key in vals.keys():
        l = vals[key]
        l = [re.sub(regex, '', x) for x in l]
        l = [x.split(' ') for x in l if 'The' not in x]
        l = [ [x for x in strings if x] for strings in l]
        df = pd.DataFrame(l)
        df.columns = ['algo', 'fold', 'rec', 'prec', 'f1']
        df = df.astype({ 'fold': 'int32', 'rec': 'float32', 'prec': 'float32', 'f1': 'float32'})
        print(df.groupby(by='algo').mean().drop(columns='fold'))
    



