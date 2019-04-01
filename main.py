import yaml
import multiprocessing           as mp
import project.data.preprocess   as pp
import project.models.embeddings.models as emb
from datetime import datetime




format 		= lambda str_ : '[' + str(datetime.now().strftime("%d/%m/%y %H:%M:%S")) + '] ' + str_
printlog    = lambda x: print(format(x))

if __name__ == '__main__':
    printlog('Starting the process, reading the configuration file.')
    conf = yaml.safe_load(open('config.yml'))
    mp.set_start_method('spawn')

    pp.preprocess(conf['evaluation']['dataset'], conf['session']['interval'])

    emb.model_runner(conf['evaluation']['dataset'], conf['models']['music2vec'])