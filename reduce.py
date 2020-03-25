import pandas as pd
from numpy.random import shuffle
from numpy import array_split
xiami 		= pd.read_csv('dataset/xiami/session_listening_history.csv', sep=',')
# m4a 		= pd.read_csv('dataset/music2all/FULL/session_listening_history.csv', sep=',')

m4a_g 	= xiami.groupby(by='session')[['song']].agg(list)
m4a_sb1	= m4a_g[m4a_g['song'].apply(len) >= 10]

m4a_sb1['len'] 	= m4a_sb1.song.apply(len)
m4a_sb1['s_id']	= m4a_sb1.index

# len_sid = m4a_sb1.groupby('len').agg(list)[['s_id']]

# lens 		= len_sid.index.values
# sessions 	= len_sid.s_id.values

# to_keep		  	= []
# remove 			= True

# for llen, session in list(zip(lens, sessions)):
# 	if len(session) > 1:
# 		shuffle(session)
# 		keep, _ = array_split(session, 2)
# 		to_keep.extend(keep)
# 	else:
# 		if remove:
# 			remove = False
# 		else:
# 			to_keep.extend(session)
# 			remove = True


print(xiami.shape)
filtered_m4a = xiami[xiami.session.isin(m4a_sb1.index)]
# print(filtered_m4a)
print(filtered_m4a.shape)

filtered_m4a.to_csv('dataset/xiami/session_listening_history.csv', sep=',', index=None)
