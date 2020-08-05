# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:49:51 2019

@author: User
"""

import os
import pandas as pd

__file__ = os.getcwd()
path_data_train = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'preprocess', 'train_preprocess_2.csv')
path_label = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'raw', 'train_label.csv')

X_train = pd.read_csv(path_data_train, engine = 'python')
y_train = pd.read_csv(path_label, engine = 'python')

y_train['leave'] = 0
y_train['leave'][y_train.survival_time < 64] = 1

y_train['amount'] = 0
y_train['amount'][y_train.amount_spent > 0] = 1

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.25, random_state = 42)
for train_index, test_index in split.split(y_train, y_train['leave']):
    strat_y_train = y_train.loc[train_index]
    strat_y_test = y_train.loc[test_index]
    
y_train = strat_y_train.sort_values(['acc_id'], ascending = True)
y_test = strat_y_test.sort_values(['acc_id'], ascending = True)


### 모델링 ###
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib

# 이탈 여부
rf_leave_clf = RandomForestClassifier(random_state = 42, max_features = 15, n_estimators = 750, n_jobs = -1)
rf_leave_clf.fit(X_train.iloc[:, 1:], y_train['leave'])
joblib.dump(rf_leave_clf, './rf_leave_clf.pkl')

xgb_leave_clf = XGBClassifier(random_state = 42, learning_rate = 0.1, gamma = 3, max_depth = 14, n_jobs = -1)
xgb_leave_clf.fit(X_train.iloc[:, 1:], y_train['leave'])
joblib.dump(xgb_leave_clf, './xgb_leave_clf.pkl')

ensemble_leave_clf = VotingClassifier(
    estimators = [('rf', rf_leave_clf), ('xgb', xgb_leave_clf)],
    voting = 'soft', n_jobs = -1)
ensemble_leave_clf.fit(X_train.iloc[:, 1:], y_train['leave'])
joblib.dump(ensemble_leave_clf, './ensemble_leave_clf.pkl')


# 결제 여부
rf_amount_clf = RandomForestClassifier(random_state = 42, max_features = 11, n_estimators = 750, n_jobs = -1)
rf_amount_clf.fit(X_train.iloc[:, 1:], y_train['amount'])
joblib.dump(rf_amount_clf, './rf_amount_clf.pkl')

xgb_amount_clf = XGBClassifier(random_state = 42, learning_rate = 0.1, gamma = 3, max_depth = 10, n_jobs = -1)
xgb_amount_clf.fit(X_train.iloc[:, 1:], y_train['amount'])
joblib.dump(xgb_amount_clf, './xgb_amount_clf.pkl')

ensemble_amount_clf = VotingClassifier(
    estimators = [('rf', rf_amount_clf), ('xgb', xgb_amount_clf)],
    voting = 'soft', n_jobs = -1)
ensemble_amount_clf.fit(X_train.iloc[:, 1:], y_train['amount'])
joblib.dump(ensemble_amount_clf, './ensemble_amount_clf.pkl')


# 추가 생존 기간
train = pd.merge(X_train, y_train, how = 'left', on = 'acc_id')

train_leave = train[train['leave'] == 1]
train_amount = train[(train['leave'] == 1) & (train['amount'] == 1)]

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.externals import joblib

rf_leave_reg = RandomForestRegressor(random_state = 42, max_features = 18, n_estimators = 750, n_jobs = -1)
rf_leave_reg.fit(train_leave.iloc[:, 1:-4], train_leave['survival_time'])
joblib.dump(rf_leave_reg, './rf_leave_reg.pkl')

xgb_leave_reg = XGBRegressor(random_state = 42, learning_rate = 0.1, gamma = 5, max_depth = 8, n_jobs = -1)
xgb_leave_reg.fit(train_leave.iloc[:, 1:-4], train_leave['survival_time'])
joblib.dump(xgb_leave_reg, './xgb_leave_reg.pkl')

extree_leave_reg = ExtraTreesRegressor(random_state = 42, bootstrap = True, max_features = 22, n_estimators = 750, n_jobs = -1)
extree_leave_reg.fit(train_leave.iloc[:, 1:-4], train_leave['survival_time'])
joblib.dump(extree_leave_reg, './extree_leave_reg.pkl')

lr_leave_reg = ElasticNet(random_state = 42, alpha = 0.1, l1_ratio = 0.8)
lr_leave_reg.fit(train_leave.iloc[:, 1:-4], train_leave['survival_time'])
joblib.dump(lr_leave_reg, './lr_leave_reg.pkl')


# 일 평균 결제 금액
rf_amount_reg = RandomForestRegressor(random_state = 42, max_features = 7, n_estimators = 500, n_jobs = -1)
rf_amount_reg.fit(train_amount.iloc[:, 1:-4], train_amount['amount_spent'])
joblib.dump(rf_amount_reg, './rf_amount_reg.pkl')

xgb_amount_reg = XGBRegressor(random_state = 42, learning_rate = 0.01, gamma = 3, max_depth = 2, n_jobs = -1)
xgb_amount_reg.fit(train_amount.iloc[:, 1:-4], train_amount['amount_spent'])
joblib.dump(xgb_amount_reg, './xgb_amount_reg.pkl')

extree_amount_reg = ExtraTreesRegressor(random_state = 42, bootstrap = True, max_features = 7, n_estimators = 300, n_jobs = -1)
extree_amount_reg.fit(train_amount.iloc[:, 1:-4], train_amount['amount_spent'])
joblib.dump(extree_amount_reg, './extree_amount_reg.pkl')

lr_amount_reg = ElasticNet(random_state = 42, alpha = 0.1, l1_ratio = 0.2)
lr_amount_reg.fit(train_amount.iloc[:, 1:-4], train_amount['amount_spent'])
joblib.dump(lr_amount_reg, './lr_amount_reg.pkl')