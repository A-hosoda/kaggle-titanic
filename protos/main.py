import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk('../input/'):
	for filename in filenames:
		print(os.path.join(dirname, filename))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')

print(train.shape)
print(test.shape)
print(gender_submission.shape)

print(train.columns)
print('-'*10)
print(test.columns)

print('-'*10)
print(train.info())

print('-'*10)
print(test.info())

print('-'*10)
print(train.head())

print('-'*10)
print(train.isnull().sum())

print('-'*10)
print(test.isnull().sum())

df_full = pd.concat([train, test], axis=0, sort=False)
print('-'*10)
print(df_full.shape)
print(df_full.describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9]))
print('-'*10)
print(df_full.describe(include = 'O'))

#import pandas_profiling as pdp
#pdp.ProfileReport(train)
