import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk('../input/'):
	for filename in filenames:
		print(os.path.join(dirname, filename))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')

train = pd.get_dummies(train, columns=['Sex', 'Embarked'])
test = pd.get_dummies(test, columns=['Sex', 'Embarked'])

train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

print(train.head())


x_train = train.drop(['Survived'], axis=1)
y_train = train['Survived']

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.33, random_state=0)

lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(valid_x, valid_y)

lgbm_params = {'objective': 'binary'}

evals_result = {}
gbm = lgb.train(params=lgbm_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=20, evals_result=evals_result, verbose_eval=10);

oof = (gbm.predict(valid_x) > 0.5).astype(int)
print('score', round(accuracy_score(valid_y, oof)*100,2))

import matplotlib.pyplot as plt

plt.plot(evals_result['training']['binary_logloss'], label='train_loss')
plt.plot(evals_result['valid_1']['binary_logloss'], label='valid_loss')
plt.legend()
