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
        
class Results():
    def __init__(self, setups, k):
        self.metrics    = {}
        self.k          = k
        for _, params, _ in setups:
            self.metrics[params] = {    'm2vTN': AlgoMetrics('m2vTN', params), 'sm2vTN': AlgoMetrics('sm2vTN', params),
                                        'csm2vTN': AlgoMetrics('csm2vTN', params), 'csm2vUK': AlgoMetrics('csm2vUK', params)}
        self.final_df = pd.DataFrame()
        
    def fold_results(self, params, m2vTN, sm2vTN, csm2vTN, csm2vUK, fold):
        self.metrics[params]['m2vTN'].add(m2vTN)
        self.metrics[params]['sm2vTN'].add(sm2vTN)
        self.metrics[params]['csm2vTN'].add(csm2vTN)
        self.metrics[params]['csm2vUK'].add(csm2vUK)
        metrics = np.vstack([m2vTN, sm2vTN, csm2vTN, csm2vUK])
        print()
        data = {    
            'params': [params] * 4,
            'algo': ['m2vTN','sm2vTN','csm2vTN','csm2vUK'],
            'folds':[fold] * 4,
            'prec': metrics[:,0],
            'rec': metrics[:,1],
            'f1': metrics[:,2]
        }
        df = pd.DataFrame(data)
        return df
    
    def get_results(self, params):
        results = self.metrics[params]
        finals  = []
        for algo in ['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK']:
            data = {    'algo'  : [algo] * (self.k + 2),
                        'folds' : [str(x) for x in range(1, self.k + 1)] + ['mean', 'std'],
                        'prec'  : results[algo].to_a('prec'),
                        'rec'   : results[algo].to_a('rec'),
                        'f1'    : results[algo].to_a('f1')  }
            df = pd.DataFrame(data)
            finals.append(df)
        return finals
    
    def concat_summary_dataframe(self, df):
        self.final_df = pd.concat([self.final_df, df])

    def setup_finish(self, params):
        m2vTN, sm2vTN, csm2vTN, csm2vUK = self.get_results(params)
        self.metrics[params] = {'m2vTN': m2vTN, 'sm2vTN': sm2vTN, 'csm2vTN': csm2vTN, 'csm2vUK': csm2vUK}
        r1 = m2vTN[m2vTN['folds'] == 'mean']
        r2 = sm2vTN[sm2vTN['folds'] == 'mean']
        r3 = csm2vTN[csm2vTN['folds'] == 'mean']
        r4 = csm2vUK[csm2vUK['folds'] == 'mean']
        df = pd.concat([r1, r2, r3, r4], axis=0)
        df = df.drop(columns=['folds']).reset_index(drop=True)
        df['params'] = np.array([params] * 4)
        self.concat_summary_dataframe(df)
        
    def report(self, path_s, path_f):
        df_full = pd.DataFrame()
        for param, metrics in self.metrics.items():
            r1 = metrics['m2vTN']
            r2 = metrics['sm2vTN']
            r3 = metrics['csm2vTN']
            r4 = metrics['csm2vUK']
            df = pd.concat([r1, r2, r3, r4])
            df['params'] = 7 * 4 * [param]
            df_full = pd.concat([df_full, df])
        
        self.final_df.set_index('params').to_csv(path_s, sep=';', decimal=',')
        df_full.set_index(['params', 'algo']).to_csv(path_f, sep=';', decimal=',')