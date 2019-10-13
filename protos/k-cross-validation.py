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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=3, shuffle=True)

score_list=[]
models=[]

for fold_, (train_index, valid_index) in enumerate(kf.split(x_train, y_train)):
	print(f'fold{fold_ + 1} start')
	train_x = x_train.iloc[train_index]
	valid_x = x_train.iloc[valid_index]
	train_y = y_train[train_index]
	valid_y = y_train[valid_index]

	lgb_train = lgb.Dataset(train_x, train_y)
	lgb_valid = lgb.Dataset(valid_x, valid_y)

	lgbm_params = {'objective': 'binary'}

	gbm = lgb.train(params = lgbm_params, train_set = lgb_train, valid_sets = [lgb_train, lgb_valid], early_stopping_rounds = 20, verbose_eval = -1)

	oof = (gbm.predict(valid_x) > 0.5).astype(int)
	score_list.append(round(accuracy_score(valid_y, oof)*100, 2))
	models.append(gbm)
	print(f'fold{fold_ + 1} end \n')

print(score_list, '平均score', round(np.mean(score_list), 2))

