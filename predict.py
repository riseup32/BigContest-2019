# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:57:00 2019

@author: User
"""

import os
import pandas as pd
import numpy as np

__file__ = os.getcwd()
path_data_test1 = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'preprocess', 'test1_preprocess_2.csv')
path_data_test2 = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'preprocess', 'test2_preprocess_2.csv')

X_test1 = pd.read_csv(path_data_test1, engine = 'python')
X_test2 = pd.read_csv(path_data_test2, engine = 'python')

path_model_ensemble_leave_clf = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'ensemble_leave_clf.pkl')

path_model_ensemble_amount_clf = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'ensemble_amount_clf.pkl')

path_model_rf_leave_reg = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'rf_leave_reg.pkl')
path_model_xgb_leave_reg = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'xgb_leave_reg.pkl')
path_model_extree_leave_reg = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'extree_leave_reg.pkl')
path_model_lr_leave_reg = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'lr_leave_reg.pkl')

path_model_rf_amount_reg = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'rf_amount_reg.pkl')
path_model_xgb_amount_reg = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'xgb_amount_reg.pkl')
path_model_extree_amount_reg = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'extree_amount_reg.pkl')
path_model_lr_amount_reg = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'model', 'lr_amount_reg.pkl')


from sklearn.externals import joblib

ensemble_leave_clf = joblib.load(path_model_ensemble_leave_clf)

ensemble_amount_clf = joblib.load(path_model_ensemble_amount_clf)

rf_leave_reg = joblib.load(path_model_rf_leave_reg)
xgb_leave_reg = joblib.load(path_model_xgb_leave_reg)
extree_leave_reg = joblib.load(path_model_extree_leave_reg)
lr_leave_reg = joblib.load(path_model_lr_leave_reg)

rf_amount_reg = joblib.load(path_model_rf_amount_reg)
xgb_amount_reg = joblib.load(path_model_xgb_amount_reg)
extree_amount_reg = joblib.load(path_model_extree_amount_reg)
lr_amount_reg = joblib.load(path_model_lr_amount_reg)


def predict(model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, data,
            w1, w2, w3, w4, w5, w6, w7, w8):
    '''
    model1 : 생존 여부 예측(classifier)
    model2 : 결제 여부 예측(classifier)
    model3~6 : 추가 생존 기간 예측(regressor)
    model7~10 : 일 평균 결제 금액 예측(regressor)
    data : test data
    w1~w4 : 추가 생존 기간 예측 모델 weight
    w5~w8 : 일 평균 결제 금액 예측 모델 weight
    '''
    
    pred1 = model1.predict(data.iloc[:, 1:])
    pred2 = model2.predict(data.iloc[:, 1:])
    pred3 = ((w1 * model3.predict(data.iloc[:, 1:])) + (w2 * model4.predict(data.iloc[:, 1:]))
            + (w3 * model5.predict(data.iloc[:, 1:])) + (w4 * model6.predict(data.iloc[:, 1:])))
    pred4 = ((w5 * model7.predict(data.iloc[:, 1:])) + (w6 * model8.predict(data.iloc[:, 1:]))
            + (w7 * model9.predict(data.iloc[:, 1:])) + (w8 * model10.predict(data.iloc[:, 1:])))
    
    pred3[pred1 == 0] = 64
    pred4[pred2 == 0] = 0
    pred3[pred3 <= 1] = 1
    pred3[pred3 >= 64] = 64
    pred4[pred4 <= 0] = 0
    
    array = np.concatenate([data.iloc[:, 0].values.reshape(-1, 1), pred3.reshape(-1, 1), pred4.reshape(-1, 1)], axis = 1)
    df = pd.DataFrame(array)
    df.columns = ['acc_id', 'survival_time', 'amount_spent']
    df['acc_id'] = df['acc_id'].astype('int32')
    df['survival_time'] = round(df['survival_time']).astype('int32')
    
    return df

test1_pred = predict(model1 = ensemble_leave_clf, model2 = ensemble_amount_clf, 
                     model3 = rf_leave_reg, model4 = xgb_leave_reg,
                     model5 = extree_leave_reg, model6 = lr_leave_reg,
                     model7 = rf_amount_reg, model8 = xgb_amount_reg,
                     model9 = extree_amount_reg, model10 = lr_amount_reg,
                     data = X_test1,
                     w1 = 0.0000, w2 = 0.4174, w3 = 0.1453, w4 = 0.0000, w5 = 1.5000, w6 = 0.7745, w7 = 1.5000, w8 = 1.5000)

test2_pred = predict(model1 = ensemble_leave_clf, model2 = ensemble_amount_clf, 
                     model3 = rf_leave_reg, model4 = xgb_leave_reg,
                     model5 = extree_leave_reg, model6 = lr_leave_reg,
                     model7 = rf_amount_reg, model8 = xgb_amount_reg,
                     model9 = extree_amount_reg, model10 = lr_amount_reg,
                     data = X_test2,
                     w1 = 0.0000, w2 = 0.4174, w3 = 0.1453, w4 = 0.0000, w5 = 1.5000, w6 = 0.7745, w7 = 1.5000, w8 = 1.5000)

test1_pred.to_csv('./test1_predict.csv', index = False)
test2_pred.to_csv('./test2_predict.csv', index = False)