import pandas as pd
import numpy as np

class AlgoMetrics():
    def __init__(self, algo, params):
        self.prec   = []
        self.rec    = []
        self.f1     = []
        self.algo   = algo
        self.params  = params
    
    def add(self, arr):
        prec, rec, f1 = arr
        self.prec.append(prec)
        self.rec.append(rec)
        self.f1.append(f1)

    def to_a(self, m_t):
        if m_t == 'prec':
            return self.prec + [np.mean(self.prec, axis=0)] + [np.std(self.prec, axis=0)]
        if m_t == 'rec':
            return self.rec + [np.mean(self.rec, axis=0)] + [np.std(self.rec, axis=0)]
        if m_t == 'f1':
            return self.f1 + [np.mean(self.f1, axis=0)] + [np.std(self.f1, axis=0)]
        return 1
        
def center(df):
    df.style.set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]}])
    return df

class Results():
    def __init__(self, setups, k):
        self.metrics    = {}
        self.k          = k
        for _, params, _ in setups:
            self.metrics[params] = {    'm2vTN': AlgoMetrics('m2vTN', params), 'sm2vTN': AlgoMetrics('sm2vTN', params),
                                        'csm2vTN': AlgoMetrics('csm2vTN', params), 'csm2vUK': AlgoMetrics('csm2vUK', params)}
        
    def fold_results(self, params, m2vTN, sm2vTN, csm2vTN, csm2vUK):
        self.metrics[params]['m2vTN'].add(m2vTN)
        self.metrics[params]['sm2vTN'].add(sm2vTN)
        self.metrics[params]['csm2vTN'].add(csm2vTN)
        self.metrics[params]['csm2vUK'].add(csm2vUK)
        return True
    
    def get_results(self, params):
        results = self.metrics[params]
        finals = []
        for algo in ['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK']:
            data = {    'algo'  : [algo] * (self.k + 2),
                        'folds' : [str(x) for x in range(1, self.k + 1)] + ['mean', 'std'],
                        'prec'  : results[algo].to_a('prec'),
                        'rec'   : results[algo].to_a('rec'),
                        'f1'    : results[algo].to_a('f1')  }
            df = pd.DataFrame(data)
            finals.append(df)
        return finals
            
    def finish(self, params):
        m2vTN, sm2vTN, csm2vTN, csm2vUK = self.get_results(params)
        self.metrics[params] = {'m2vTN': m2vTN, 'sm2vTN': sm2vTN, 'csm2vTN': csm2vTN, 'csm2vUK': csm2vUK}
        print(center(m2vTN[m2vTN['folds'] == 'mean']))
        print(center(sm2vTN[sm2vTN['folds'] == 'mean']))
        print(center(csm2vTN[csm2vTN['folds'] == 'mean']))
        print(center(csm2vUK[csm2vUK['folds'] == 'mean']))