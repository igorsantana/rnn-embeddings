import pandas as pd
import numpy as np

       
class Results():
    def __init__(self, setups, k):
        self.metrics    = {}
        self.k          = k
        self.final_df = pd.DataFrame()
        
    def fold_results(self, params, m2vTN, sm2vTN, csm2vTN, csm2vUK, fold):
        metrics = np.vstack([m2vTN, sm2vTN, csm2vTN, csm2vUK])
        print()
        data = {    
            'params': [params] * 4,
            'algo': ['m2vTN','sm2vTN','csm2vTN','csm2vUK'],
            'folds':[fold] * 4,
            'prec': metrics[:,0],
            'rec': metrics[:,1],
            'f1': metrics[:,2],
            'map': metrics[:,3],
            'ndcg@5': metrics[:,4],
            'p@5': metrics[:,5]
        }
        df = pd.DataFrame(data)
        return df
    