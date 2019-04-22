import pandas as pd

v = [['a', 'c', 'q'], ['b'], ['b']]

df = pd.DataFrame([[0,1,2,3,4,'a'], [5,6,7,8,9, 'b']], columns=['v', 'w', 'x', 'y', 'z', 'index']).set_index('index')

qq = lambda x: sum([True for y in v if x in y])

df['z3'] = [qq(x) for x in df.index.values]
df['z'] += df['z3']

print(df.loc[:,'z'])
