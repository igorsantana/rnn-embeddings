class Setups():
    def __init__(self, config):
        self.__config = config
        self.models_config = config['models']

    def get_config(self):
        return self.__config

    def rnn_setups(self):
        c = self.models_config['rnn']
        
        for m in c['model']:
            for w in c['window']:
                for n in c['num_units']:
                    for e in c['embedding_dim']:
                        for ep in c['epochs']:
                                for bi in c['bi']:
                                    yield { 'window': int(w), 'model': m, 'dim': int(e), 'batch': int(c['batch']), 
                                            'epochs': int(ep), 'num_units': int(n), 'bidi': bi}
                            
    def d2v_m2v_setups(self, model):
        c = self.models_config[model]
        for w in c['window']:
            for sample in c['negative_sample']:
                for down in c['down_sample']:
                    for lr in c['learning_rate']:
                        for ep in c['epochs']:
                            for dim in c['embedding_dim']:
                                yield { 'window': w, 'dim': int(dim), 'lr': float(lr), 'down': float(down), 'epochs': int(ep),  'neg_sample': float(sample)}

    def glove_setups(self):
        c = self.models_config['glove']
        for w in c['window']:
            for dim in c['embedding_dim']:
                for lr in c['learning_rate']:
                    for ep in c['epochs']:
                        yield { 'window': int(w), 'dim': int(dim), 'lr': float(lr), 'epochs': int(ep)}

    def __return_gen(self, model):
        if model == 'rnn':
            return self.rnn_setups()
        if model == 'music2vec' or model == 'doc2vec':
            return self.d2v_m2v_setups(model)
        if model == 'glove':
            return self.glove_setups()

    def get_generators(self):
        generators = []
        for emb_methods in self.__config['embeddings'].items():
            k, v = emb_methods
            if v['usage'] == True:
                generators.append((k, self.__return_gen(k)))
        return generators
    
    def setup_to_string(self, id, setup_obj, model_type):
        setup_str = '--'.join([x + ':' + str(y) for x,y in list(setup_obj.items())])
        return '{}--{}--{}'.format(model_type, id, setup_str)