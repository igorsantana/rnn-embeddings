from os import listdir
import yaml


paths= [('music2all_20', 'tmp/music2all_20/models/'), ('music2all_40', 'tmp/music2all_40/models/'), ('music2all_60', 'tmp/music2all_60/models/')]


# config = yaml.safe_load(open('config.yml', 'r'))

# config['embeddings']['music2vec']['usage'] = False
# for ds, path in paths:
#     config['evaluation']['dataset'] = ds
#     config['embeddings']['rnn']['usage'] = True
#     for f in listdir(path):
#         if f.endswith('.csv'):
#             config['embeddings']['rnn']['path'] = f
#             config['logfile'] = 'outputs/' + f.split('.csv')[0] + '.log'
#             yaml.dump(config, open('configs/'+ f.split('.csv')[0] + '.yml', 'w'), default_flow_style=False)


data = []
for x in listdir('configs'):
    data.append('python main.py --config=configs/' + x )


print(' && '.join(data))


