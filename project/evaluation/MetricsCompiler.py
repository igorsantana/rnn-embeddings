import pandas as pd

class MetricsCompiler:

    def __init__(self):
            self.prec                    = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
            self.rec                     = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
            self.fmeas                   = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])
            self.hitrate                 = pd.DataFrame([], index=[0,1,2,3,4], columns=['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK'])


    def add_metric(self, metric, fold, algo, value):
        if metric == 'prec':
            self.prec.loc[fold, algo] = value
        if metric == 'rec':
            self.rec.loc[fold, algo] = value
        if metric == 'fmeas':
            self.fmeas.loc[fold, algo] = value
        if metric == 'hitrate':
            self.hitrate.loc[fold, algo] = value