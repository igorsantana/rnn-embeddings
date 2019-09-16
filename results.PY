from os import listdir
import pandas as pd


algos = ['m2vTN', 'sm2vTN', 'csm2vTN', 'csm2vUK']

df_all = pd.DataFrame({}, columns=['emb', 'seq', 'm2vTN_F1', 'sm2vTN_F1', 'csm2vTN_F1', 'csm2vUK_F1'])

for f in listdir('outputs'):
    content = open('outputs/' + f, 'r').read()
    content = content.split('\n')
    content = [line.split(']')[1:][0] for line in content if len(line) > 0]
    content = [[w for w in line.split(' ') if w] for line in content]
    df = pd.DataFrame(content[1:])
    df.columns  = content[0]
    df['Prec']  = df['Prec'].astype(float)
    df['Rec']   = df['Rec'].astype(float)
    df['F1']    = df['F1'].astype(float)
    data = [f.split('.log')[0]]
    data.append(f.split('.log')[0].split('_')[-1].split('-')[-1])
    
    for algo in algos:
        prec = round(df[df['Algo'] == algo]['Prec'].mean(), 4)
        rec  = round(df[df['Algo'] == algo]['Rec'].mean(), 4)
        f1   = round(df[df['Algo'] == algo]['F1'].mean(), 4)
        data.append(f1)

    print(data)
    df_all = df_all.append(pd.Series(data, index=df_all.columns), ignore_index=True)

df_all.sort_values(by='emb', ascending=False).to_csv('results.csv', index=None, sep=';', decimal=',')
    
