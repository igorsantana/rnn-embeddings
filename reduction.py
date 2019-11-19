import pandas as pd
import numpy as np
from numpy.random import shuffle
from math import floor
from pathlib import Path
pwd = 'dataset/'
ds  = 'xiami'
df  = pd.read_csv(pwd + ds + '/session_listening_history.csv')

small_ds    = ('xiami-small', 0.1)
smaller_ds  = ('xiami-smaller', 0.01)

users       = df.user.unique()
no_users    = df.user.nunique()

small_ds_no_users   = floor(small_ds[1] * no_users)
smaller_ds_no_users = floor(smaller_ds[1] * no_users)

shuffle(users)
small_users = users[:small_ds_no_users]
shuffle(users)
smaller_users = users[:smaller_ds_no_users]

Path(pwd + small_ds[0]).mkdir(exist_ok=True)

df[df.user.isin(small_users)].to_csv(pwd + small_ds[0] + '/session_listening_history.csv', sep=',', index=None)

Path(pwd + smaller_ds[0]).mkdir(exist_ok=True)

df[df.user.isin(smaller_users)].to_csv(pwd + smaller_ds[0] + '/session_listening_history.csv', sep=',', index=None)
