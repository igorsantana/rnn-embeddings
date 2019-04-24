import pandas as pd

v = [['a', 'c', 'q'], ['b'], ['b']]

df = pd.DataFrame([[0,1,2,3,0.12312,'a'], [5,6,7,8,0.9412, 'b']], columns=['v', 'w', 'x', 'y', 'cos', 'index']).set_index('index')

qq = lambda x: sum([True for y in v if x in y])

df['size'] = [qq(x) for x in df.index.values]

# SIMILARITIES BETWEEN USER u AND THE MOST SIMILAR users
q = [0.5, 0.2, 0.3, 0.1]

# SIZE = NUMBER OF USERS WHO LISTENED TO THE MUSIC
# COS  = COSINE SIMILARITY FROM USER TO MUSIC
# SIM  = SIMILARITY BETWEEN USER u AND USER v

df['pref'] = 1

print(df.loc[:,'pref'])

for x in q:
  df['pref'] += (x / (df.loc[:,'size'] + df.loc[:,'cos']))
  print(df.loc[:,'pref'])

