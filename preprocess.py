# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:30:41 2019

@author: User
"""

import os
import pandas as pd
import numpy as np

__file__ = os.getcwd()
path = os.path.join(os.path.realpath(__file__).rsplit('\\', 1)[0], 'raw')

path_train_activity = os.path.join(path, 'train_activity.csv')
path_train_combat = os.path.join(path, 'train_combat.csv')
path_train_pledge = os.path.join(path, 'train_pledge.csv')
path_train_payment = os.path.join(path, 'train_payment.csv')
path_train_trade = os.path.join(path, 'train_trade.csv')
path_train_label = os.path.join(path, 'train_label.csv')

train1 = pd.read_csv(path_train_activity, engine = 'python')
train2 = pd.read_csv(path_train_combat, engine = 'python')
train3 = pd.read_csv(path_train_pledge, engine = 'python')
train4 = pd.read_csv(path_train_payment, engine = 'python')
train5 = pd.read_csv(path_train_trade, engine = 'python')
y_train = pd.read_csv(path_train_label, engine = 'python')

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

train_id = y_train['acc_id'].values
test_id = y_test['acc_id'].values

train_id.sort()
test_id.sort()


### 전처리 ###
def divide_by_week(data, variable):
    '''
    데이터와 변수를 넣으면 각 유저 아이디에 대한 주차별 변수의 합을 구함
    '''
    
    data_sub = data.groupby(['acc_id', 'day'])[variable].sum().unstack('day').fillna(0)
    week_1 = data_sub.iloc[:, 0:7].sum(axis = 1).values
    week_2 = data_sub.iloc[:, 7:14].sum(axis = 1).values
    week_3 = data_sub.iloc[:, 14:21].sum(axis = 1).values
    week_4 = data_sub.iloc[:, 21:28].sum(axis = 1).values
    print(variable, 'done')
        
    return week_1, week_2, week_3, week_4


def pledge_divide_by_week(data, variable):
    '''
    데이터와 변수를 넣으면 각 혈맹 아이디에 대한 주차별 변수의 평균을 구함
    '''
    
    data_sub = data.groupby(['pledge_id', 'day'])[variable].mean().unstack('day').fillna(0)
    week_1 = data_sub.iloc[:, 0:7].mean(axis = 1).values
    week_2 = data_sub.iloc[:, 7:14].mean(axis = 1).values
    week_3 = data_sub.iloc[:, 14:21].mean(axis = 1).values
    week_4 = data_sub.iloc[:, 21:28].mean(axis = 1).values
    print(variable, 'done')
        
    return week_1, week_2, week_3, week_4


def char_divide_by_week(data, variable):
    '''
    데이터와 변수를 넣으면 각 캐릭터 아이디의 유저에 대한 주차별 변수의 평균을 구함
    '''
    
    data_sub = data.groupby(['acc_id', 'day'])[variable].mean().unstack('day').fillna(0)
    week_1 = data_sub.iloc[:, 0:7].mean(axis = 1).values
    week_2 = data_sub.iloc[:, 7:14].mean(axis = 1).values
    week_3 = data_sub.iloc[:, 14:21].mean(axis = 1).values
    week_4 = data_sub.iloc[:, 21:28].mean(axis = 1).values
    print(variable, 'done')
        
    return week_1, week_2, week_3, week_4


def preprocess(data1, data2, data3, data4, data5, user_id, train = True):
    '''
    유저와 케릭터에 대한 28일 동안의 모든 활동 데이터를 각 유저에 대한 정보로 전처리
    data1 : 기본 활동 데이터(DataFrame)
    data2 : 전투 데이터(DataFrame)
    data3 : 혈맹 데이터(DataFrame)
    data4 : 결제 데이터(DataFrame)
    data5 : 거래 데이터(DataFrame)
    user_id(=acc_id) : 유저별 id(np.array)
    train : default = True, train data set에 대해서만 이상치 대체
    '''
    
    # user_id에 해당하는 행만 subset
    index_list = []
    for i in range(0, len(data1)):
        if(data1['acc_id'][i] in user_id):
            index_list.append(i)            
    data1_sub = data1.loc[index_list]
    
    index_list = []
    for i in range(0, len(data2)):
        if(data2['acc_id'][i] in user_id):
            index_list.append(i)            
    data2_sub = data2.loc[index_list]
    
    index_list = []
    for i in range(0, len(data3)):
        if(data3['acc_id'][i] in user_id):
            index_list.append(i)            
    data3_sub = data3.loc[index_list]
    
    index_list = []
    for i in range(0, len(data4)):
        if(data4['acc_id'][i] in user_id):
            index_list.append(i)            
    data4_sub = data4.loc[index_list]
    
    index_list = []
    for i in range(0, len(data5)):
        if(data5['source_acc_id'][i] in user_id):
            index_list.append(i)      
    source_data = data5.loc[index_list]
    
    index_list = []
    for i in range(0, len(data5)):
        if(data5['target_acc_id'][i] in user_id):
            index_list.append(i)      
    target_data = data5.loc[index_list]
    
    
    # 연속형 변수명만 추출
    data1_col = data1.columns[4:]
    data2_col = data2.columns[6:]
    data3_col = data3.columns[5:-1]
    
    
    # 각 유저에 대한 연속형 변수를 주차별 합으로 새로운 변수 만듦
    dic = {}  # 새 변수를 담기 위한 빈 dictionary

    for var_name in data1_col:
        week_1, week_2, week_3, week_4 = divide_by_week(data = data1_sub, variable = var_name)
        key_1 = var_name + '_1'
        key_2 = var_name + '_2'
        key_3 = var_name + '_3'
        key_4 = var_name + '_4'
        dic[key_1] = week_1
        dic[key_2] = week_2
        dic[key_3] = week_3
        dic[key_4] = week_4
        
    for var_name in data2_col:
        week_1, week_2, week_3, week_4 = divide_by_week(data = data2_sub, variable = var_name)
        key_1 = 'combat_' + var_name + '_1'
        key_2 = 'combat_' + var_name + '_2'
        key_3 = 'combat_' + var_name + '_3'
        key_4 = 'combat_' + var_name + '_4'
        dic[key_1] = week_1
        dic[key_2] = week_2
        dic[key_3] = week_3
        dic[key_4] = week_4
    
    
    # user_id와 연속형 변수들을 concatenate 하여 data_array로 저장
    user_id_clone = user_id.copy()
    data_array = user_id_clone.reshape(-1, 1)

    for key in dic.keys():
        data_array = np.concatenate([data_array, dic[key].reshape(-1, 1)], axis = 1)
        
        
    # data_array를 데이터프레임 형태로 바꿔주고 컬럼명 지정
    colnames = ['acc_id']
    colnames.extend(list(dic.keys())[0:])

    data = pd.DataFrame(data_array)
    data.columns = colnames
    
    
    # 혈맹 데이터를 이용해 혈맹 군집화
    dic = {}
    char_list = []
    char_matrix = data3_sub.groupby(['acc_id', 'char_id']).count()['day']
    data3_id = data3_sub['acc_id'].unique()
    data3_id.sort()
    for i in data3_id:
        char_list.append(np.argmax(char_matrix[i]))
    char_id = np.array(char_list)

    index_list = []
    for i in range(0, len(data3_sub)):
        if(data3_sub['char_id'].iloc[i] in char_id):
            index_list.append(i)            
    data3_sub_2 = data3_sub.iloc[index_list]

    for var_name in data3_col:
        week_1, week_2, week_3, week_4 = char_divide_by_week(data = data3_sub_2, variable = var_name)
        key_1 = var_name + '_1'
        key_2 = var_name + '_2'
        key_3 = var_name + '_3'
        key_4 = var_name + '_4'
        dic[key_1] = week_1
        dic[key_2] = week_2
        dic[key_3] = week_3
        dic[key_4] = week_4

    data3_id_clone = data3_id.copy()
    char_data_array = np.concatenate([data3_id_clone.reshape(-1, 1), char_id.reshape(-1, 1)], axis = 1)

    for key in dic.keys():
        char_data_array = np.concatenate([char_data_array, dic[key].reshape(-1, 1)], axis = 1)

    colnames = ['acc_id', 'char_id']
    colnames.extend(list(dic.keys())[0:])

    char_data = pd.DataFrame(char_data_array)
    char_data.columns = colnames
    char_data = char_data.fillna(0)
    
    if(train == True):
        # random_attacker_cnt
        data3_sub_day_1 = data3_sub[(data3_sub['day'] == 1) | (data3_sub['day'] == 8)]
        data3_sub_day_2 = data3_sub[(data3_sub['day'] == 2) | (data3_sub['day'] == 9)]
        data3_sub_day_3 = data3_sub[(data3_sub['day'] == 3) | (data3_sub['day'] == 10)]
        data3_sub_day_4 = data3_sub[(data3_sub['day'] == 4) | (data3_sub['day'] == 11)]
        data3_sub_day_5 = data3_sub[(data3_sub['day'] == 5) | (data3_sub['day'] == 12)]
        data3_sub_day_6 = data3_sub[(data3_sub['day'] == 6) | (data3_sub['day'] == 13)]
        data3_sub_day_7 = data3_sub[(data3_sub['day'] == 7) | (data3_sub['day'] == 14)]

        data3_sub_day_6_mean = data3_sub_day_6.groupby('char_id')['random_attacker_cnt'].mean()
        data3_sub_day_6_mean = pd.DataFrame(data3_sub_day_6_mean)
        data3_sub_day_6_mean.columns = ['random_attacker_cnt_new']
        val6 = pd.merge(data3_sub[data3_sub['day'] == 20], data3_sub_day_6_mean, how = 'left', on = 'char_id')
        val6 = val6.fillna(0)

        data3_sub_day_7_mean = data3_sub_day_7.groupby('char_id')['random_attacker_cnt'].mean()
        data3_sub_day_7_mean = pd.DataFrame(data3_sub_day_7_mean)
        data3_sub_day_7_mean.columns = ['random_attacker_cnt_new']
        val7 = pd.merge(data3_sub[data3_sub['day'] == 21], data3_sub_day_7_mean, how = 'left', on = 'char_id')
        val7 = val7.fillna(0)

        val1 = data3_sub[data3_sub['day'] == 15]
        val2 = data3_sub[data3_sub['day'] == 16]
        val3 = data3_sub[data3_sub['day'] == 17]
        val4 = data3_sub[data3_sub['day'] == 18]
        val5 = data3_sub[data3_sub['day'] == 19]
        val = pd.concat([val1, val2, val3, val4, val5, val6, val7], axis = 0, ignore_index = True)

        val['random_attacker_cnt_new'][val['random_attacker_cnt_new'].isna()] = val['random_attacker_cnt'][val['random_attacker_cnt_new'].isna()]
        random_attacker_cnt_3_new = val.groupby('char_id')['random_attacker_cnt_new'].mean()
        random_attacker_cnt_3_new = pd.DataFrame(random_attacker_cnt_3_new)
        random_attacker_cnt_3_new.columns = ['random_attacker_cnt_new']
        char_data = pd.merge(char_data, random_attacker_cnt_3_new, how = 'left', on = 'char_id')
        char_data['random_attacker_cnt_3'][~char_data['random_attacker_cnt_new'].isna()] = char_data['random_attacker_cnt_new'][~char_data['random_attacker_cnt_new'].isna()]
        char_data = char_data.drop('random_attacker_cnt_new', axis = 1)


        data3_sub_day_1_mean = data3_sub_day_1.groupby('char_id')['random_attacker_cnt'].mean()
        data3_sub_day_1_mean = pd.DataFrame(data3_sub_day_1_mean)
        data3_sub_day_1_mean.columns = ['random_attacker_cnt_new']
        val1 = pd.merge(data3_sub[data3_sub['day'] == 22], data3_sub_day_1_mean, how = 'left', on = 'char_id')
        val1 = val1.fillna(0)

        val2 = data3_sub[data3_sub['day'] == 23]
        val3 = data3_sub[data3_sub['day'] == 24]
        val4 = data3_sub[data3_sub['day'] == 25]
        val5 = data3_sub[data3_sub['day'] == 26]
        val6 = data3_sub[data3_sub['day'] == 27]
        val7 = data3_sub[data3_sub['day'] == 28]
        val = pd.concat([val1, val2, val3, val4, val5, val6, val7], axis = 0, ignore_index = True)

        val['random_attacker_cnt_new'][val['random_attacker_cnt_new'].isna()] = val['random_attacker_cnt'][val['random_attacker_cnt_new'].isna()]
        random_attacker_cnt_4_new = val.groupby('char_id')['random_attacker_cnt_new'].mean()
        random_attacker_cnt_4_new = pd.DataFrame(random_attacker_cnt_4_new)
        random_attacker_cnt_4_new.columns = ['random_attacker_cnt_new']
        char_data = pd.merge(char_data, random_attacker_cnt_4_new, how = 'left', on = 'char_id')
        char_data['random_attacker_cnt_4'][~char_data['random_attacker_cnt_new'].isna()] = char_data['random_attacker_cnt_new'][~char_data['random_attacker_cnt_new'].isna()]
        char_data = char_data.drop('random_attacker_cnt_new', axis = 1)

        # same_pledge_cnt
        char_data['same_pledge_cnt_1'] = (char_data['same_pledge_cnt_2'] + char_data['same_pledge_cnt_3'] + char_data['same_pledge_cnt_4']) / 3

        # temp_cnt
        data3_sub_day_1 = data3_sub[(data3_sub['day'] == 1) | (data3_sub['day'] == 8) | (data3_sub['day'] == 15)]
        data3_sub_day_2 = data3_sub[(data3_sub['day'] == 2) | (data3_sub['day'] == 9) | (data3_sub['day'] == 16)]
        data3_sub_day_3 = data3_sub[(data3_sub['day'] == 3) | (data3_sub['day'] == 10) | (data3_sub['day'] == 17)]
        data3_sub_day_4 = data3_sub[(data3_sub['day'] == 4) | (data3_sub['day'] == 11) | (data3_sub['day'] == 18)]
        data3_sub_day_5 = data3_sub[(data3_sub['day'] == 5) | (data3_sub['day'] == 12) | (data3_sub['day'] == 19)]
        data3_sub_day_6 = data3_sub[(data3_sub['day'] == 6) | (data3_sub['day'] == 13) | (data3_sub['day'] == 20)]
        data3_sub_day_7 = data3_sub[(data3_sub['day'] == 7) | (data3_sub['day'] == 14) | (data3_sub['day'] == 21)]

        data3_sub_day_2_mean = data3_sub_day_2.groupby('char_id')['temp_cnt'].mean()
        data3_sub_day_2_mean = pd.DataFrame(data3_sub_day_2_mean)
        data3_sub_day_2_mean.columns = ['temp_cnt_new']
        val2 = pd.merge(data3_sub[data3_sub['day'] == 23], data3_sub_day_2_mean, how = 'left', on = 'char_id')
        val2 = val2.fillna(0)

        data3_sub_day_3_mean = data3_sub_day_3.groupby('char_id')['temp_cnt'].mean()
        data3_sub_day_3_mean = pd.DataFrame(data3_sub_day_3_mean)
        data3_sub_day_3_mean.columns = ['temp_cnt_new']
        val3 = pd.merge(data3_sub[data3_sub['day'] == 24], data3_sub_day_3_mean, how = 'left', on = 'char_id')
        val3 = val3.fillna(0)

        data3_sub_day_4_mean = data3_sub_day_4.groupby('char_id')['temp_cnt'].mean()
        data3_sub_day_4_mean = pd.DataFrame(data3_sub_day_4_mean)
        data3_sub_day_4_mean.columns = ['temp_cnt_new']
        val4 = pd.merge(data3_sub[data3_sub['day'] == 25], data3_sub_day_4_mean, how = 'left', on = 'char_id')
        val4 = val4.fillna(0)

        data3_sub_day_5_mean = data3_sub_day_5.groupby('char_id')['temp_cnt'].mean()
        data3_sub_day_5_mean = pd.DataFrame(data3_sub_day_5_mean)
        data3_sub_day_5_mean.columns = ['temp_cnt_new']
        val5 = pd.merge(data3_sub[data3_sub['day'] == 26], data3_sub_day_5_mean, how = 'left', on = 'char_id')
        val5 = val5.fillna(0)

        data3_sub_day_6_mean = data3_sub_day_6.groupby('char_id')['temp_cnt'].mean()
        data3_sub_day_6_mean = pd.DataFrame(data3_sub_day_6_mean)
        data3_sub_day_6_mean.columns = ['temp_cnt_new']
        val6 = pd.merge(data3_sub[data3_sub['day'] == 27], data3_sub_day_6_mean, how = 'left', on = 'char_id')
        val6 = val6.fillna(0)

        data3_sub_day_7_mean = data3_sub_day_7.groupby('char_id')['temp_cnt'].mean()
        data3_sub_day_7_mean = pd.DataFrame(data3_sub_day_7_mean)
        data3_sub_day_7_mean.columns = ['temp_cnt_new']
        val7 = pd.merge(data3_sub[data3_sub['day'] == 28], data3_sub_day_7_mean, how = 'left', on = 'char_id')
        val7 = val7.fillna(0)

        val1 = data3_sub[data3_sub['day'] == 22]
        val = pd.concat([val1, val2, val3, val4, val5, val6, val7], axis = 0, ignore_index = True)

        val['temp_cnt_new'][val['temp_cnt_new'].isna()] = val['temp_cnt'][val['temp_cnt_new'].isna()]
        temp_cnt_4_new = val.groupby('char_id')['temp_cnt_new'].mean()
        temp_cnt_4_new = pd.DataFrame(temp_cnt_4_new)
        temp_cnt_4_new.columns = ['temp_cnt_new']
        char_data = pd.merge(char_data, temp_cnt_4_new, how = 'left', on = 'char_id')
        char_data['temp_cnt_4'][~char_data['temp_cnt_new'].isna()] = char_data['temp_cnt_new'][~char_data['temp_cnt_new'].isna()]
        char_data = char_data.drop('temp_cnt_new', axis = 1)
    
    
    char_data['pledge_group'] = kmean.predict(char_data.iloc[:, 2:]) + 1
    data = pd.merge(data, char_data.loc[:, ['acc_id', 'pledge_group']], how = 'left', on = 'acc_id')
    data = data.fillna(0)
    data['pledge_group'] = data['pledge_group'].astype('int32')
    
    
    # 결제 데이터에서 28일 동안 결제 횟수, 1회 결제시 결제 금액 추출    
    amount_cnt = data4_sub.groupby('acc_id')['amount_spent'].count()
    amount_mean = data4_sub.groupby('acc_id')['amount_spent'].sum() / data4_sub.groupby('acc_id')['amount_spent'].count()
    acc_id_sorted = np.sort(data4_sub['acc_id'].unique())
    
    amount_array = np.concatenate([acc_id_sorted.reshape(-1, 1), amount_cnt.values.reshape(-1, 1),
                                  amount_mean.values.reshape(-1, 1)], axis = 1)
    amount_data = pd.DataFrame(amount_array)
    amount_data.columns = ['acc_id', 'amount_cnt', 'amount_mean']
    
    data = pd.merge(data, amount_data, how = 'left', on = 'acc_id')
    data = data.fillna(0)  # 결제 기록이 없어 nan인 값을 모두 0으로 대체
    
    
    # 기본 활동 데이터의 서버를 가장 플레이 시간이 많은 케릭터 기준으로 추출
    char_list = []
    char_matrix = data1_sub.groupby(['acc_id', 'char_id'])['playtime'].sum()
    for i in user_id:
        char_list.append(np.argmax(char_matrix[i]))
        
    server_list = []
    for i in char_list:
        user_char = data1_sub[data1_sub['char_id'] == i]
        char_server = user_char['server'].values[-1]
        server_list.append(char_server)
    
    server_array = np.concatenate([np.array(user_id).reshape(-1, 1), np.array(server_list).reshape(-1, 1)], axis = 1)
    server_data = pd.DataFrame(server_array)
    server_data.columns = ['acc_id', 'server']
    server_data['acc_id'] = server_data['acc_id'].astype('float64')
    data = pd.merge(data, server_data, how = 'left', on = 'acc_id')
    
    # 0은 특수 서버, 1은 일반 서버, 2는 non pvp 서버
    data['server_group'] = 0
    data['server_group'][(data['server'] == 'aa') | (data['server'] == 'ab') | (data['server'] == 'ac') |
                         (data['server'] == 'ad') | (data['server'] == 'ae') | (data['server'] == 'af') |
                         (data['server'] == 'ag') | (data['server'] == 'ah') | (data['server'] == 'ai') |
                         (data['server'] == 'aj') | (data['server'] == 'ak') | (data['server'] == 'al') |
                         (data['server'] == 'am') | (data['server'] == 'an') | (data['server'] == 'ao') |
                         (data['server'] == 'ap') | (data['server'] == 'aq') | (data['server'] == 'ar') |
                         (data['server'] == 'as') | (data['server'] == 'at') | (data['server'] == 'au') |
                         (data['server'] == 'av') | (data['server'] == 'aw') | (data['server'] == 'ax') |
                         (data['server'] == 'ay') | (data['server'] == 'az') | (data['server'] == 'ba') |
                         (data['server'] == 'bb') | (data['server'] == 'bc') | (data['server'] == 'bd')] = 1
    data['server_group'][(data['server'] == 'bi') | (data['server'] == 'bs') | (data['server'] == 'bg')] = 2
    data = data.drop('server', axis = 1)
    
    
    # 전투 데이터의 직업, 레벨 변수를 가장 접속일 수가 많은 케릭터 기준으로 추출    
    char_list = []
    char_matrix = data2_sub.groupby(['acc_id', 'char_id']).count()['day']
    for i in user_id:
        char_list.append(np.argmax(char_matrix[i]))
        
    class_list = []
    level_list = []
    for i in char_list:
        user_char = data2_sub[data2_sub['char_id'] == i]
        char_class = user_char['class'].values[-1]
        char_level = user_char['level'].values[-1]
        class_list.append(char_class)
        level_list.append(char_level)
    
    char_array = np.concatenate([np.array(user_id).reshape(-1, 1), np.array(class_list).reshape(-1, 1),
                                 np.array(level_list).reshape(-1, 1)], axis = 1)
    char_data = pd.DataFrame(char_array)
    char_data.columns = ['acc_id', 'class', 'level']
    char_data['acc_id'] = char_data['acc_id'].astype('float64')
    data = pd.merge(char_data, data, how = 'left', on = 'acc_id')
    
    
    # 범주형 변수 one-hot encoding
    server_group_dummies = pd.get_dummies(data['server_group']).rename(columns = lambda x : 'server_group_' + str(x))
    class_dummies = pd.get_dummies(data['class']).rename(columns = lambda x : 'class_' + str(x))
    pledge_group_dummies = pd.get_dummies(data['pledge_group']).rename(columns = lambda x : 'pledge_group_' + str(x))

    data = pd.concat([data, server_group_dummies, class_dummies, pledge_group_dummies], axis = 1)
    data.drop(['server_group', 'class', 'pledge_group'], axis = 1, inplace = True)
    
    
    # 거래 데이터를 user_id에 대하여 판매, 구매로 구분
    trade_data = pd.DataFrame(user_id)
    trade_data.columns = ['acc_id']
    
    for i in range(1, 5):
        source_data_week = source_data[(source_data['day'] >= (7 * i - 6)) & (source_data['day'] <= (7 * i))]
        source_data_week_sum = source_data_week.groupby(['source_acc_id', 'item_type'])['item_price'].sum().unstack('item_type')
        source_data_week_sum = source_data_week_sum.fillna(0)
        source_data_week_sum = source_data_week_sum.reset_index(level = 'source_acc_id')
        source_data_week_sum = source_data_week_sum.rename(columns = {'source_acc_id' : 'acc_id'})
        col = 'source_price_'+ source_data_week_sum.columns[1:] + '_' + str(i)
        source_data_week_sum.columns = source_data_week_sum.columns[0:1].append(col)
        trade_data = pd.merge(trade_data, source_data_week_sum, how = 'left', on = 'acc_id')
        
    for i in range(1, 5):
        source_data_week = source_data[(source_data['day'] >= (7 * i - 6)) & (source_data['day'] <= (7 * i))]
        source_data_week_sum = source_data_week.groupby(['source_acc_id', 'item_type'])['item_amount'].sum().unstack('item_type')
        source_data_week_sum = source_data_week_sum.fillna(0)
        source_data_week_sum = source_data_week_sum.reset_index(level = 'source_acc_id')
        source_data_week_sum = source_data_week_sum.rename(columns = {'source_acc_id' : 'acc_id'})
        col = 'source_amount_'+ source_data_week_sum.columns[1:] + '_' + str(i)
        source_data_week_sum.columns = source_data_week_sum.columns[0:1].append(col)
        trade_data = pd.merge(trade_data, source_data_week_sum, how = 'left', on = 'acc_id')
    
    for i in range(1, 5):
        target_data_week = target_data[(target_data['day'] >= (7 * i - 6)) & (target_data['day'] <= (7 * i))]
        target_data_week_sum = target_data_week.groupby(['target_acc_id', 'item_type'])['item_price'].sum().unstack('item_type')
        target_data_week_sum = target_data_week_sum.fillna(0)
        target_data_week_sum = target_data_week_sum.reset_index(level = 'target_acc_id')
        target_data_week_sum = target_data_week_sum.rename(columns = {'target_acc_id' : 'acc_id'})
        col = 'target_price_'+ target_data_week_sum.columns[1:] + '_' + str(i)
        target_data_week_sum.columns = target_data_week_sum.columns[0:1].append(col)
        trade_data = pd.merge(trade_data, target_data_week_sum, how = 'left', on = 'acc_id')
        
    for i in range(1, 5):
        target_data_week = target_data[(target_data['day'] >= (7 * i - 6)) & (target_data['day'] <= (7 * i))]
        target_data_week_sum = target_data_week.groupby(['target_acc_id', 'item_type'])['item_amount'].sum().unstack('item_type')
        target_data_week_sum = target_data_week_sum.fillna(0)
        target_data_week_sum = target_data_week_sum.reset_index(level = 'target_acc_id')
        target_data_week_sum = target_data_week_sum.rename(columns = {'target_acc_id' : 'acc_id'})
        col = 'target_amount_'+ target_data_week_sum.columns[1:] + '_' + str(i)
        target_data_week_sum.columns = target_data_week_sum.columns[0:1].append(col)
        trade_data = pd.merge(trade_data, target_data_week_sum, how = 'left', on = 'acc_id')
        
    trade_data = trade_data.fillna(0)
    
    
    # 주요 거래 아이템(아데나, 기타) 외 아이템은 28일 총합
    main_item = ['adena', 'etc']
    item_list = []
    for item in main_item:
        for i in range(0, len(trade_data.columns)):
            if(item in trade_data.columns[i]):
                item_list.append(i)                
    trade_data_sub = trade_data.iloc[:, item_list]
    
    sub_item = ['weapon', 'armor', 'accessory', 'spell', 'enchant_scroll']
    for item in sub_item:
        item_list = []
        for i in range(0, len(trade_data.columns)):
            if(item in trade_data.columns[i]):
                item_list.append(i) 
        colname1 = 'source_price_' + item
        colname2 = 'source_amount_' + item
        colname3 = 'target_price_' + item
        colname4 = 'target_amount_' + item
        trade_data_sub[colname1] = trade_data.iloc[:, item_list].iloc[:, :4].sum(axis = 1)
        trade_data_sub[colname2] = trade_data.iloc[:, item_list].iloc[:, 4:8].sum(axis = 1)
        trade_data_sub[colname3] = trade_data.iloc[:, item_list].iloc[:, 8:12].sum(axis = 1)
        trade_data_sub[colname4] = trade_data.iloc[:, item_list].iloc[:, 12:].sum(axis = 1)
    
    
    trade_data_sub['acc_id'] = trade_data['acc_id']
    
    data = pd.merge(data, trade_data_sub, how = 'left', on = 'acc_id')
        
    
    return data


index_list = []
for i in range(0, len(train3)):
    if(train3['acc_id'][i] in train_id):
        index_list.append(i)            
train3_sub = train3.loc[index_list]

dic = {}
train3_col = train3_sub.columns[5:-1]

for var_name in train3_col:
    week_1, week_2, week_3, week_4 = pledge_divide_by_week(data = train3_sub, variable = var_name)
    key_1 = var_name + '_1'
    key_2 = var_name + '_2'
    key_3 = var_name + '_3'
    key_4 = var_name + '_4'
    dic[key_1] = week_1
    dic[key_2] = week_2
    dic[key_3] = week_3
    dic[key_4] = week_4
    
pledge_id = train3_sub['pledge_id'].unique()
pledge_id.sort()
pledge_id_clone = pledge_id.copy()
pledge_data_array = pledge_id_clone.reshape(-1, 1)

for key in dic.keys():
    pledge_data_array = np.concatenate([pledge_data_array, dic[key].reshape(-1, 1)], axis = 1)
        
colnames = ['pledge_id']
colnames.extend(list(dic.keys())[0:])

pledge_data = pd.DataFrame(pledge_data_array)
pledge_data.columns = colnames
pledge_data = pledge_data.fillna(0)


train3_sub_day_1 = train3_sub[(train3_sub['day'] == 1) | (train3_sub['day'] == 8)]
train3_sub_day_2 = train3_sub[(train3_sub['day'] == 2) | (train3_sub['day'] == 9)]
train3_sub_day_3 = train3_sub[(train3_sub['day'] == 3) | (train3_sub['day'] == 10)]
train3_sub_day_4 = train3_sub[(train3_sub['day'] == 4) | (train3_sub['day'] == 11)]
train3_sub_day_5 = train3_sub[(train3_sub['day'] == 5) | (train3_sub['day'] == 12)]
train3_sub_day_6 = train3_sub[(train3_sub['day'] == 6) | (train3_sub['day'] == 13)]
train3_sub_day_7 = train3_sub[(train3_sub['day'] == 7) | (train3_sub['day'] == 14)]

train3_sub_day_6_mean = train3_sub_day_6.groupby('pledge_id')['random_attacker_cnt'].mean()
train3_sub_day_6_mean = pd.DataFrame(train3_sub_day_6_mean)
train3_sub_day_6_mean.columns = ['random_attacker_cnt_new']
val6 = pd.merge(train3_sub[train3_sub['day'] == 20], train3_sub_day_6_mean, how = 'left', on = 'pledge_id')
val6 = val6.fillna(0)

train3_sub_day_7_mean = train3_sub_day_7.groupby('pledge_id')['random_attacker_cnt'].mean()
train3_sub_day_7_mean = pd.DataFrame(train3_sub_day_7_mean)
train3_sub_day_7_mean.columns = ['random_attacker_cnt_new']
val7 = pd.merge(train3_sub[train3_sub['day'] == 21], train3_sub_day_7_mean, how = 'left', on = 'pledge_id')
val7 = val7.fillna(0)

val1 = train3_sub[train3_sub['day'] == 15]
val2 = train3_sub[train3_sub['day'] == 16]
val3 = train3_sub[train3_sub['day'] == 17]
val4 = train3_sub[train3_sub['day'] == 18]
val5 = train3_sub[train3_sub['day'] == 19]
val = pd.concat([val1, val2, val3, val4, val5, val6, val7], axis = 0, ignore_index = True)

val['random_attacker_cnt_new'][val['random_attacker_cnt_new'].isna()] = val['random_attacker_cnt'][val['random_attacker_cnt_new'].isna()]
random_attacker_cnt_3_new = val.groupby('pledge_id')['random_attacker_cnt_new'].mean()
random_attacker_cnt_3_new = pd.DataFrame(random_attacker_cnt_3_new)
random_attacker_cnt_3_new.columns = ['random_attacker_cnt_new']
pledge_data = pd.merge(pledge_data, random_attacker_cnt_3_new, how = 'left', on = 'pledge_id')
pledge_data['random_attacker_cnt_3'][~pledge_data['random_attacker_cnt_new'].isna()] = pledge_data['random_attacker_cnt_new'][~pledge_data['random_attacker_cnt_new'].isna()]
pledge_data = pledge_data.drop('random_attacker_cnt_new', axis = 1)

train3_sub_day_1_mean = train3_sub_day_1.groupby('pledge_id')['random_attacker_cnt'].mean()
train3_sub_day_1_mean = pd.DataFrame(train3_sub_day_1_mean)
train3_sub_day_1_mean.columns = ['random_attacker_cnt_new']
val1 = pd.merge(train3_sub[train3_sub['day'] == 22], train3_sub_day_1_mean, how = 'left', on = 'pledge_id')
val1 = val1.fillna(0)

val2 = train3_sub[train3_sub['day'] == 23]
val3 = train3_sub[train3_sub['day'] == 24]
val4 = train3_sub[train3_sub['day'] == 25]
val5 = train3_sub[train3_sub['day'] == 26]
val6 = train3_sub[train3_sub['day'] == 27]
val7 = train3_sub[train3_sub['day'] == 28]
val = pd.concat([val1, val2, val3, val4, val5, val6, val7], axis = 0, ignore_index = True)

val['random_attacker_cnt_new'][val['random_attacker_cnt_new'].isna()] = val['random_attacker_cnt'][val['random_attacker_cnt_new'].isna()]
random_attacker_cnt_4_new = val.groupby('pledge_id')['random_attacker_cnt_new'].mean()
random_attacker_cnt_4_new = pd.DataFrame(random_attacker_cnt_4_new)
random_attacker_cnt_4_new.columns = ['random_attacker_cnt_new']
pledge_data = pd.merge(pledge_data, random_attacker_cnt_4_new, how = 'left', on = 'pledge_id')
pledge_data['random_attacker_cnt_4'][~pledge_data['random_attacker_cnt_new'].isna()] = pledge_data['random_attacker_cnt_new'][~pledge_data['random_attacker_cnt_new'].isna()]
pledge_data = pledge_data.drop('random_attacker_cnt_new', axis = 1)

pledge_data['same_pledge_cnt_1'] = (pledge_data['same_pledge_cnt_2'] + pledge_data['same_pledge_cnt_3'] + pledge_data['same_pledge_cnt_4']) / 3

train3_sub_day_1 = train3_sub[(train3_sub['day'] == 1) | (train3_sub['day'] == 8) | (train3_sub['day'] == 15)]
train3_sub_day_2 = train3_sub[(train3_sub['day'] == 2) | (train3_sub['day'] == 9) | (train3_sub['day'] == 16)]
train3_sub_day_3 = train3_sub[(train3_sub['day'] == 3) | (train3_sub['day'] == 10) | (train3_sub['day'] == 17)]
train3_sub_day_4 = train3_sub[(train3_sub['day'] == 4) | (train3_sub['day'] == 11) | (train3_sub['day'] == 18)]
train3_sub_day_5 = train3_sub[(train3_sub['day'] == 5) | (train3_sub['day'] == 12) | (train3_sub['day'] == 19)]
train3_sub_day_6 = train3_sub[(train3_sub['day'] == 6) | (train3_sub['day'] == 13) | (train3_sub['day'] == 20)]
train3_sub_day_7 = train3_sub[(train3_sub['day'] == 7) | (train3_sub['day'] == 14) | (train3_sub['day'] == 21)]

train3_sub_day_2_mean = train3_sub_day_2.groupby('pledge_id')['temp_cnt'].mean()
train3_sub_day_2_mean = pd.DataFrame(train3_sub_day_2_mean)
train3_sub_day_2_mean.columns = ['temp_cnt_new']
val2 = pd.merge(train3_sub[train3_sub['day'] == 23], train3_sub_day_2_mean, how = 'left', on = 'pledge_id')
val2 = val2.fillna(0)

train3_sub_day_3_mean = train3_sub_day_3.groupby('pledge_id')['temp_cnt'].mean()
train3_sub_day_3_mean = pd.DataFrame(train3_sub_day_3_mean)
train3_sub_day_3_mean.columns = ['temp_cnt_new']
val3 = pd.merge(train3_sub[train3_sub['day'] == 24], train3_sub_day_3_mean, how = 'left', on = 'pledge_id')
val3 = val3.fillna(0)

train3_sub_day_4_mean = train3_sub_day_4.groupby('pledge_id')['temp_cnt'].mean()
train3_sub_day_4_mean = pd.DataFrame(train3_sub_day_4_mean)
train3_sub_day_4_mean.columns = ['temp_cnt_new']
val4 = pd.merge(train3_sub[train3_sub['day'] == 25], train3_sub_day_4_mean, how = 'left', on = 'pledge_id')
val4 = val4.fillna(0)

train3_sub_day_5_mean = train3_sub_day_5.groupby('pledge_id')['temp_cnt'].mean()
train3_sub_day_5_mean = pd.DataFrame(train3_sub_day_5_mean)
train3_sub_day_5_mean.columns = ['temp_cnt_new']
val5 = pd.merge(train3_sub[train3_sub['day'] == 26], train3_sub_day_5_mean, how = 'left', on = 'pledge_id')
val5 = val5.fillna(0)

train3_sub_day_6_mean = train3_sub_day_6.groupby('pledge_id')['temp_cnt'].mean()
train3_sub_day_6_mean = pd.DataFrame(train3_sub_day_6_mean)
train3_sub_day_6_mean.columns = ['temp_cnt_new']
val6 = pd.merge(train3_sub[train3_sub['day'] == 27], train3_sub_day_6_mean, how = 'left', on = 'pledge_id')
val6 = val6.fillna(0)

train3_sub_day_7_mean = train3_sub_day_7.groupby('pledge_id')['temp_cnt'].mean()
train3_sub_day_7_mean = pd.DataFrame(train3_sub_day_7_mean)
train3_sub_day_7_mean.columns = ['temp_cnt_new']
val7 = pd.merge(train3_sub[train3_sub['day'] == 28], train3_sub_day_7_mean, how = 'left', on = 'pledge_id')
val7 = val7.fillna(0)

val1 = train3_sub[train3_sub['day'] == 22]
val = pd.concat([val1, val2, val3, val4, val5, val6, val7], axis = 0, ignore_index = True)

val['temp_cnt_new'][val['temp_cnt_new'].isna()] = val['temp_cnt'][val['temp_cnt_new'].isna()]
temp_cnt_4_new = val.groupby('pledge_id')['temp_cnt_new'].mean()
temp_cnt_4_new = pd.DataFrame(temp_cnt_4_new)
temp_cnt_4_new.columns = ['temp_cnt_new']
pledge_data = pd.merge(pledge_data, temp_cnt_4_new, how = 'left', on = 'pledge_id')
pledge_data['temp_cnt_4'][~pledge_data['temp_cnt_new'].isna()] = pledge_data['temp_cnt_new'][~pledge_data['temp_cnt_new'].isna()]
pledge_data = pledge_data.drop('temp_cnt_new', axis = 1)

from sklearn.cluster import KMeans

X = pledge_data.iloc[:, 1:]
kmean = KMeans(n_clusters = 3, random_state = 42, n_jobs = -1)
kmean.fit(X)


X_train = preprocess(train1, train2, train3, train4, train5, train_id, train = True)
X_test = preprocess(train1, train2, train3, train4, train5, test_id, train = True)

X_train.to_csv('./train_preprocess_1.csv', index = False)
X_test.to_csv('./test_preprocess_1.csv', index = False)


path_test1_activity = os.path.join(path, 'test1_activity.csv')
path_test1_combat = os.path.join(path, 'test1_combat.csv')
path_test1_pledge = os.path.join(path, 'test1_pledge.csv')
path_test1_payment = os.path.join(path, 'test1_payment.csv')
path_test1_trade = os.path.join(path, 'test1_trade.csv')

test1_1 = pd.read_csv(path_test1_activity, engine = 'python')
test1_2 = pd.read_csv(path_test1_combat, engine = 'python')
test1_3 = pd.read_csv(path_test1_pledge, engine = 'python')
test1_4 = pd.read_csv(path_test1_payment, engine = 'python')
test1_5 = pd.read_csv(path_test1_trade, engine = 'python')

test1_id = test1_1['acc_id'].unique()
test1_id.sort()

X_test1 = preprocess(test1_1, test1_2, test1_3, test1_4, test1_5, test1_id, train = False)

path_test2_activity = os.path.join(path, 'test2_activity.csv')
path_test2_combat = os.path.join(path, 'test2_combat.csv')
path_test2_pledge = os.path.join(path, 'test2_pledge.csv')
path_test2_payment = os.path.join(path, 'test2_payment.csv')
path_test2_trade = os.path.join(path, 'test2_trade.csv')

test2_1 = pd.read_csv(path_test2_activity, engine = 'python')
test2_2 = pd.read_csv(path_test2_combat, engine = 'python')
test2_3 = pd.read_csv(path_test2_pledge, engine = 'python')
test2_4 = pd.read_csv(path_test2_payment, engine = 'python')
test2_5 = pd.read_csv(path_test2_trade, engine = 'python')

test2_id = test2_1['acc_id'].unique()
test2_id.sort()

X_test2 = preprocess(test2_1, test2_2, test2_3, test2_4, test2_5, test2_id, train = False)

X_test1.to_csv('./test1_preprocess_1.csv', index = False)
X_test2.to_csv('./test2_preprocess_1.csv', index = False)



### 추가 전처리 ###
X_train = pd.read_csv('./train_preprocess_1.csv')
X_test = pd.read_csv('./test_preprocess_1.csv')
X_test1 = pd.read_csv('./test1_preprocess_1.csv')
X_test2 = pd.read_csv('./test2_preprocess_1.csv')

var_list = []
for i in range(0, len(X_train.columns)):
    if('class' in X_train.columns[i]):
        var_list.append(i)
        
X_train = X_train.drop(X_train.columns[var_list], axis = 1)
X_test = X_test.drop(X_test.columns[var_list], axis = 1)
X_test1 = X_test1.drop(X_test1.columns[var_list], axis = 1)
X_test2 = X_test2.drop(X_test2.columns[var_list], axis = 1)

X_train['playtime_fluctuation'] = (X_train['playtime_4'] + 1) / (X_train['playtime_3'] + 1)
X_test['playtime_fluctuation'] = (X_test['playtime_4'] + 1) / (X_test['playtime_3'] + 1)
X_test1['playtime_fluctuation'] = (X_test1['playtime_4'] + 1) / (X_test1['playtime_3'] + 1)
X_test2['playtime_fluctuation'] = (X_test2['playtime_4'] + 1) / (X_test2['playtime_3'] + 1)

index_list = []
for i in range(0, len(train1)):
    if(train1['acc_id'][i] in train_id):
        index_list.append(i)            
train1_sub = train1.loc[index_list]

index_list = []
for i in range(0, len(train1)):
    if(train1['acc_id'][i] in test_id):
        index_list.append(i)            
test1_sub = train1.loc[index_list]

index_list = []
for i in range(0, len(test1_1)):
    if(test1_1['acc_id'][i] in test1_id):
        index_list.append(i)            
test1_1_sub = test1_1.loc[index_list]

index_list = []
for i in range(0, len(test2_1)):
    if(test2_1['acc_id'][i] in test2_id):
        index_list.append(i)            
test2_1_sub = test2_1.loc[index_list]

def count_num(data, variable, user_id, version = 1):
    '''
    유저 아이디 별로 원하는 변수에 대한 unique값 count
    원하는 user_id에 대한 sub data가 있는 경우 version = 1, 없는 경우 version = 2
    version = 2 의 경우 subset 과정으로 시간이 더 오래 소요
    '''
    
    if(version == 1):
        data_sub = data
    else:
        index_list = []
        for i in range(0, len(data)):
            if(data['acc_id'][i] in user_id):
                index_list.append(i)            
        data_sub = data.loc[index_list]
    
    variable_unique = data_sub.groupby('acc_id')[variable].unique()
    variable_num = np.zeros(len(user_id))

    for i in range(0, len(user_id)):
        variable_num[i] = len(variable_unique[user_id[i]])
        
    return variable_num


day_num = count_num(data = train1_sub, variable = 'day', user_id = train_id, version = 1)
server_num = count_num(data = train1_sub, variable = 'server', user_id = train_id, version = 1)
char_num = count_num(data = train1_sub, variable = 'char_id', user_id = train_id, version = 1)

X_train['log_total'] = day_num
X_train['server_num'] = server_num
X_train['char_num'] = char_num

day_num = count_num(data = test1_sub, variable = 'day', user_id = test_id, version = 1)
server_num = count_num(data = test1_sub, variable = 'server', user_id = test_id, version = 1)
char_num = count_num(data = test1_sub, variable = 'char_id', user_id = test_id, version = 1)

X_test['log_total'] = day_num
X_test['server_num'] = server_num
X_test['char_num'] = char_num

day_num = count_num(data = test1_1_sub, variable = 'day', user_id = test1_id, version = 1)
server_num = count_num(data = test1_1_sub, variable = 'server', user_id = test1_id, version = 1)
char_num = count_num(data = test1_1_sub, variable = 'char_id', user_id = test1_id, version = 1)

X_test1['log_total'] = day_num
X_test1['server_num'] = server_num
X_test1['char_num'] = char_num

day_num = count_num(data = test2_1_sub, variable = 'day', user_id = test2_id, version = 1)
server_num = count_num(data = test2_1_sub, variable = 'server', user_id = test2_id, version = 1)
char_num = count_num(data = test2_1_sub, variable = 'char_id', user_id = test2_id, version = 1)

X_test2['log_total'] = day_num
X_test2['server_num'] = server_num
X_test2['char_num'] = char_num

X_train['npc_kill_1'] = (X_train['npc_kill_2'] + X_train['npc_kill_3'] + X_train['npc_kill_4']) / 3
X_train['solo_exp_1'] = (X_train['solo_exp_2'] + X_train['solo_exp_3'] + X_train['solo_exp_4']) / 3
X_train['rich_monster_2'] = (X_train['rich_monster_1'] + X_train['rich_monster_3'] + X_train['rich_monster_4']) / 3
X_train['combat_same_pledge_cnt_1'] = (X_train['combat_same_pledge_cnt_2'] + X_train['combat_same_pledge_cnt_3'] + X_train['combat_same_pledge_cnt_4']) / 3

X_test['npc_kill_1'] = (X_test['npc_kill_2'] + X_test['npc_kill_3'] + X_test['npc_kill_4']) / 3
X_test['solo_exp_1'] = (X_test['solo_exp_2'] + X_test['solo_exp_3'] + X_test['solo_exp_4']) / 3
X_test['rich_monster_2'] = (X_test['rich_monster_1'] + X_test['rich_monster_3'] + X_test['rich_monster_4']) / 3
X_test['combat_same_pledge_cnt_1'] = (X_test['combat_same_pledge_cnt_2'] + X_test['combat_same_pledge_cnt_3'] + X_test['combat_same_pledge_cnt_4']) / 3

train1_sub_day = train1_sub[(train1_sub['day'] == 1) | (train1_sub['day'] == 8) | (train1_sub['day'] == 22)]
test1_sub_day = test1_sub[(test1_sub['day'] == 1) | (test1_sub['day'] == 8) | (test1_sub['day'] == 22)]

val1 = train1_sub_day.groupby('acc_id')['game_money_change'].sum() / 3
val2 = train1_sub[train1_sub['day'] == 16].groupby('acc_id')['game_money_change'].sum()
val3 = train1_sub[train1_sub['day'] == 17].groupby('acc_id')['game_money_change'].sum()
val4 = train1_sub[train1_sub['day'] == 18].groupby('acc_id')['game_money_change'].sum()
val5 = train1_sub[train1_sub['day'] == 19].groupby('acc_id')['game_money_change'].sum()
val6 = train1_sub[train1_sub['day'] == 20].groupby('acc_id')['game_money_change'].sum()
val7 = train1_sub[train1_sub['day'] == 21].groupby('acc_id')['game_money_change'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(train_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
game_money_change_sum = df.iloc[:, 1:].sum(axis = 1)
X_train['game_money_change_3'] = game_money_change_sum

val1 = test1_sub_day.groupby('acc_id')['game_money_change'].sum() / 3
val2 = test1_sub[test1_sub['day'] == 16].groupby('acc_id')['game_money_change'].sum()
val3 = test1_sub[test1_sub['day'] == 17].groupby('acc_id')['game_money_change'].sum()
val4 = test1_sub[test1_sub['day'] == 18].groupby('acc_id')['game_money_change'].sum()
val5 = test1_sub[test1_sub['day'] == 19].groupby('acc_id')['game_money_change'].sum()
val6 = test1_sub[test1_sub['day'] == 20].groupby('acc_id')['game_money_change'].sum()
val7 = test1_sub[test1_sub['day'] == 21].groupby('acc_id')['game_money_change'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(test_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
game_money_change_sum = df.iloc[:, 1:].sum(axis = 1)
X_test['game_money_change_3'] = game_money_change_sum

val1 = train1_sub_day.groupby('acc_id')['enchant_count'].sum() / 3
val2 = train1_sub[train1_sub['day'] == 16].groupby('acc_id')['enchant_count'].sum()
val3 = train1_sub[train1_sub['day'] == 17].groupby('acc_id')['enchant_count'].sum()
val4 = train1_sub[train1_sub['day'] == 18].groupby('acc_id')['enchant_count'].sum()
val5 = train1_sub[train1_sub['day'] == 19].groupby('acc_id')['enchant_count'].sum()
val6 = train1_sub[train1_sub['day'] == 20].groupby('acc_id')['enchant_count'].sum()
val7 = train1_sub[train1_sub['day'] == 21].groupby('acc_id')['enchant_count'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(train_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
enchant_count_sum = df.iloc[:, 1:].sum(axis = 1)
X_train['enchant_count_3'] = enchant_count_sum

val1 = test1_sub_day.groupby('acc_id')['enchant_count'].sum() / 3
val2 = test1_sub[test1_sub['day'] == 16].groupby('acc_id')['enchant_count'].sum()
val3 = test1_sub[test1_sub['day'] == 17].groupby('acc_id')['enchant_count'].sum()
val4 = test1_sub[test1_sub['day'] == 18].groupby('acc_id')['enchant_count'].sum()
val5 = test1_sub[test1_sub['day'] == 19].groupby('acc_id')['enchant_count'].sum()
val6 = test1_sub[test1_sub['day'] == 20].groupby('acc_id')['enchant_count'].sum()
val7 = test1_sub[test1_sub['day'] == 21].groupby('acc_id')['enchant_count'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(test_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
enchant_count_sum = df.iloc[:, 1:].sum(axis = 1)
X_test['enchant_count_3'] = enchant_count_sum

index_list = []
for i in range(0, len(train2)):
    if(train2['acc_id'][i] in train_id):
        index_list.append(i)            
train2_sub = train2.loc[index_list]

index_list = []
for i in range(0, len(train2)):
    if(train2['acc_id'][i] in test_id):
        index_list.append(i)            
test2_sub = train2.loc[index_list]

index_list = []
for i in range(0, len(test1_2)):
    if(test1_2['acc_id'][i] in test1_id):
        index_list.append(i)            
test1_2_sub = test1_2.loc[index_list]

index_list = []
for i in range(0, len(test2_2)):
    if(test2_2['acc_id'][i] in test2_id):
        index_list.append(i)            
test2_2_sub = test2_2.loc[index_list]

train2_sub_day_1 = train2_sub[(train2_sub['day'] == 8) | (train2_sub['day'] == 15)]
train2_sub_day_2 = train2_sub[(train2_sub['day'] == 9) | (train2_sub['day'] == 16)]
train2_sub_day_3 = train2_sub[(train2_sub['day'] == 10) | (train2_sub['day'] == 17)]
train2_sub_day_4 = train2_sub[(train2_sub['day'] == 11) | (train2_sub['day'] == 18)]
train2_sub_day_5 = train2_sub[(train2_sub['day'] == 12) | (train2_sub['day'] == 19)]
train2_sub_day_6 = train2_sub[(train2_sub['day'] == 13) | (train2_sub['day'] == 20)]
train2_sub_day_7 = train2_sub[(train2_sub['day'] == 14) | (train2_sub['day'] == 21)]

test2_sub_day_1 = test2_sub[(test2_sub['day'] == 8) | (test2_sub['day'] == 15)]
test2_sub_day_2 = test2_sub[(test2_sub['day'] == 9) | (test2_sub['day'] == 16)]
test2_sub_day_3 = test2_sub[(test2_sub['day'] == 10) | (test2_sub['day'] == 17)]
test2_sub_day_4 = test2_sub[(test2_sub['day'] == 11) | (test2_sub['day'] == 18)]
test2_sub_day_5 = test2_sub[(test2_sub['day'] == 12) | (test2_sub['day'] == 19)]
test2_sub_day_6 = test2_sub[(test2_sub['day'] == 13) | (test2_sub['day'] == 20)]
test2_sub_day_7 = test2_sub[(test2_sub['day'] == 14) | (test2_sub['day'] == 21)]

val1 = train2_sub_day_1.groupby('acc_id')['temp_cnt'].sum() / 2
val2 = train2_sub_day_2.groupby('acc_id')['temp_cnt'].sum() / 2
val3 = train2_sub_day_3.groupby('acc_id')['temp_cnt'].sum() / 2
val4 = train2_sub[train2_sub['day'] == 4].groupby('acc_id')['temp_cnt'].sum()
val5 = train2_sub[train2_sub['day'] == 5].groupby('acc_id')['temp_cnt'].sum()
val6 = train2_sub[train2_sub['day'] == 6].groupby('acc_id')['temp_cnt'].sum()
val7 = train2_sub[train2_sub['day'] == 7].groupby('acc_id')['temp_cnt'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(train_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
combat_temp_cnt_sum = df.iloc[:, 1:].sum(axis = 1)
X_train['combat_temp_cnt_1'] = combat_temp_cnt_sum

val1 = train2_sub[train2_sub['day'] == 22].groupby('acc_id')['temp_cnt'].sum()
val2 = train2_sub_day_2.groupby('acc_id')['temp_cnt'].sum() / 2
val3 = train2_sub_day_3.groupby('acc_id')['temp_cnt'].sum() / 2
val4 = train2_sub_day_4.groupby('acc_id')['temp_cnt'].sum() / 2
val5 = train2_sub_day_5.groupby('acc_id')['temp_cnt'].sum() / 2
val6 = train2_sub_day_6.groupby('acc_id')['temp_cnt'].sum() / 2
val7 = train2_sub_day_7.groupby('acc_id')['temp_cnt'].sum() / 2

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(train_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
combat_temp_cnt_sum = df.iloc[:, 1:].sum(axis = 1)
X_train['combat_temp_cnt_4'] = combat_temp_cnt_sum

val1 = test2_sub_day_1.groupby('acc_id')['temp_cnt'].sum() / 2
val2 = test2_sub_day_2.groupby('acc_id')['temp_cnt'].sum() / 2
val3 = test2_sub_day_3.groupby('acc_id')['temp_cnt'].sum() / 2
val4 = test2_sub[test2_sub['day'] == 4].groupby('acc_id')['temp_cnt'].sum()
val5 = test2_sub[test2_sub['day'] == 5].groupby('acc_id')['temp_cnt'].sum()
val6 = test2_sub[test2_sub['day'] == 6].groupby('acc_id')['temp_cnt'].sum()
val7 = test2_sub[test2_sub['day'] == 7].groupby('acc_id')['temp_cnt'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(test_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
combat_temp_cnt_sum = df.iloc[:, 1:].sum(axis = 1)
X_test['combat_temp_cnt_1'] = combat_temp_cnt_sum

val1 = test2_sub[test2_sub['day'] == 22].groupby('acc_id')['temp_cnt'].sum()
val2 = test2_sub_day_2.groupby('acc_id')['temp_cnt'].sum() / 2
val3 = test2_sub_day_3.groupby('acc_id')['temp_cnt'].sum() / 2
val4 = test2_sub_day_4.groupby('acc_id')['temp_cnt'].sum() / 2
val5 = test2_sub_day_5.groupby('acc_id')['temp_cnt'].sum() / 2
val6 = test2_sub_day_6.groupby('acc_id')['temp_cnt'].sum() / 2
val7 = test2_sub_day_7.groupby('acc_id')['temp_cnt'].sum() / 2

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(test_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
combat_temp_cnt_sum = df.iloc[:, 1:].sum(axis = 1)
X_test['combat_temp_cnt_4'] = combat_temp_cnt_sum

index_list = []
for i in range(0, len(train5)):
    if(train5['source_acc_id'][i] in train_id):
        index_list.append(i)            
train5_source = train5.loc[index_list]

index_list = []
for i in range(0, len(train5)):
    if(train5['target_acc_id'][i] in train_id):
        index_list.append(i)            
train5_target = train5.loc[index_list]

index_list = []
for i in range(0, len(train5)):
    if(train5['source_acc_id'][i] in test_id):
        index_list.append(i)            
test5_source = train5.loc[index_list]

index_list = []
for i in range(0, len(train5)):
    if(train5['target_acc_id'][i] in test_id):
        index_list.append(i)            
test5_target = train5.loc[index_list]

train5_source = train5_source.rename(columns = {'source_acc_id' : 'acc_id'})
train5_target = train5_target.rename(columns = {'target_acc_id' : 'acc_id'})
test5_source = test5_source.rename(columns = {'source_acc_id' : 'acc_id'})
test5_target = test5_target.rename(columns = {'target_acc_id' : 'acc_id'})

X_train['source_amount_etc_3'] = (X_train['source_amount_etc_1'] + X_train['source_amount_etc_2']) / 2
X_train['target_amount_etc_3'] = (X_train['target_amount_etc_1'] + X_train['target_amount_etc_2']) / 2

X_test['source_amount_etc_3'] = (X_test['source_amount_etc_1'] + X_test['source_amount_etc_2']) / 2
X_test['target_amount_etc_3'] = (X_test['target_amount_etc_1'] + X_test['target_amount_etc_2']) / 2

train5_source_day_1 = train5_source[(train5_source['day'] == 1) | (train5_source['day'] == 8)]
train5_source_day_2 = train5_source[(train5_source['day'] == 2) | (train5_source['day'] == 9)]

test5_source_day_1 = test5_source[(test5_source['day'] == 1) | (test5_source['day'] == 8)]
test5_source_day_2 = test5_source[(test5_source['day'] == 2) | (test5_source['day'] == 9)]

val1 = train5_source_day_1[train5_source_day_1['item_type'] == 'etc'].groupby('acc_id')['item_amount'].sum() / 2
val2 = train5_source_day_2[train5_source_day_2['item_type'] == 'etc'].groupby('acc_id')['item_amount'].sum() / 2
val3 = train5_source[(train5_source['day'] == 24) & (train5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val4 = train5_source[(train5_source['day'] == 25) & (train5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val5 = train5_source[(train5_source['day'] == 26) & (train5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val6 = train5_source[(train5_source['day'] == 27) & (train5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val7 = train5_source[(train5_source['day'] == 28) & (train5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(train_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
source_amount_etc_sum = df.iloc[:, 1:].sum(axis = 1)
X_train['source_amount_etc_4'] = source_amount_etc_sum

val1 = test5_source_day_1[test5_source_day_1['item_type'] == 'etc'].groupby('acc_id')['item_amount'].sum() / 2
val2 = test5_source_day_2[test5_source_day_2['item_type'] == 'etc'].groupby('acc_id')['item_amount'].sum() / 2
val3 = test5_source[(test5_source['day'] == 24) & (test5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val4 = test5_source[(test5_source['day'] == 25) & (test5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val5 = test5_source[(test5_source['day'] == 26) & (test5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val6 = test5_source[(test5_source['day'] == 27) & (test5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val7 = test5_source[(test5_source['day'] == 28) & (test5_source['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(train_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
source_amount_etc_sum = df.iloc[:, 1:].sum(axis = 1)
X_test['source_amount_etc_4'] = source_amount_etc_sum

train5_target_day_1 = train5_target[(train5_target['day'] == 1) | (train5_target['day'] == 8)]
train5_target_day_2 = train5_target[(train5_target['day'] == 2) | (train5_target['day'] == 9)]

test5_target_day_1 = test5_target[(test5_target['day'] == 1) | (test5_target['day'] == 8)]
test5_target_day_2 = test5_target[(test5_target['day'] == 2) | (test5_target['day'] == 9)]

val1 = train5_target_day_1[train5_target_day_1['item_type'] == 'etc'].groupby('acc_id')['item_amount'].sum() / 2
val2 = train5_target_day_2[train5_target_day_2['item_type'] == 'etc'].groupby('acc_id')['item_amount'].sum() / 2
val3 = train5_target[(train5_target['day'] == 24) & (train5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val4 = train5_target[(train5_target['day'] == 25) & (train5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val5 = train5_target[(train5_target['day'] == 26) & (train5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val6 = train5_target[(train5_target['day'] == 27) & (train5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val7 = train5_target[(train5_target['day'] == 28) & (train5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(train_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
target_amount_etc_sum = df.iloc[:, 1:].sum(axis = 1)
X_train['target_amount_etc_4'] = target_amount_etc_sum

val1 = test5_target_day_1[test5_target_day_1['item_type'] == 'etc'].groupby('acc_id')['item_amount'].sum() / 2
val2 = test5_target_day_2[test5_target_day_2['item_type'] == 'etc'].groupby('acc_id')['item_amount'].sum() / 2
val3 = test5_target[(test5_target['day'] == 24) & (test5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val4 = test5_target[(test5_target['day'] == 25) & (test5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val5 = test5_target[(test5_target['day'] == 26) & (test5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val6 = test5_target[(test5_target['day'] == 27) & (test5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()
val7 = test5_target[(test5_target['day'] == 28) & (test5_target['item_type'] == 'etc')].groupby('acc_id')['item_amount'].sum()

df1 = pd.DataFrame(val1)
df2 = pd.DataFrame(val2)
df3 = pd.DataFrame(val3)
df4 = pd.DataFrame(val4)
df5 = pd.DataFrame(val5)
df6 = pd.DataFrame(val6)
df7 = pd.DataFrame(val7)

df = pd.DataFrame(train_id)
df.columns = ['acc_id']

df = pd.merge(df, df1, how = 'left', on = 'acc_id')
df = pd.merge(df, df2, how = 'left', on = 'acc_id')
df = pd.merge(df, df3, how = 'left', on = 'acc_id')
df = pd.merge(df, df4, how = 'left', on = 'acc_id')
df = pd.merge(df, df5, how = 'left', on = 'acc_id')
df = pd.merge(df, df6, how = 'left', on = 'acc_id')
df = pd.merge(df, df7, how = 'left', on = 'acc_id')

df = df.fillna(0)
target_amount_etc_sum = df.iloc[:, 1:].sum(axis = 1)
X_test['target_amount_etc_4'] = target_amount_etc_sum

price_adena = ['source_price_adena_1', 'source_price_adena_2', 'source_price_adena_3', 'source_price_adena_4',
              'target_price_adena_1', 'target_price_adena_2', 'target_price_adena_3', 'target_price_adena_4']

X_train = X_train.drop(price_adena, axis = 1)
X_test = X_test.drop(price_adena, axis = 1)
X_test1 = X_test1.drop(price_adena, axis = 1)
X_test2 = X_test2.drop(price_adena, axis = 1)

def week_to_mean(data, variable, drop = True):
    '''
    주차별 합으로 이루어진 변수를 28일 동안의 평균으로 변환
    주차별 변수는 삭제
    '''
    
    var_list = []
    for i in range(0, len(data.columns)):
        if(variable in data.columns[i]):
            var_list.append(i)
        
    column_name = variable + '_mean'
    data[column_name] = data.iloc[:, var_list].mean(axis = 1)
    if(drop == True):
        data = data.drop(data.columns[var_list], axis = 1)
    
    return data

variable_list = ['exp_recovery', 'private_shop', 'enchant_count', 'combat_pledge_cnt', 'combat_random_attacker_cnt',
                'combat_random_defender_cnt', 'combat_same_pledge_cnt', 'source_price_etc']

for variable in variable_list:
    X_train = week_to_mean(data = X_train, variable = variable)
    
for variable in variable_list:
    X_test = week_to_mean(data = X_test, variable = variable)
    
for variable in variable_list:
    X_test1 = week_to_mean(data = X_test1, variable = variable)
    
for variable in variable_list:
    X_test2 = week_to_mean(data = X_test2, variable = variable)
    
def level_diff(data, user_id):
    '''
    28일 동안의 레별 변화를 구함
    '''
    
    char_list = []
    char_matrix = data.groupby(['acc_id', 'char_id']).count()['day']
    for i in user_id:
        char_list.append(np.argmax(char_matrix[i]))
        
    level_diff_list = []
    for i in char_list:
        user_char = data[data['char_id'] == i]
        char_level_diff = user_char['level'].max() - user_char['level'].min()
        level_diff_list.append(char_level_diff)
        
    return np.array(level_diff_list)

train_level_diff = level_diff(data = train2_sub, user_id = train_id)
X_train['level_diff'] = train_level_diff

test_level_diff = level_diff(data = test2_sub, user_id = test_id)
X_test['level_diff'] = test_level_diff

test1_level_diff = level_diff(data = test1_2_sub, user_id = test1_id)
X_test1['level_diff'] = test1_level_diff

test2_level_diff = level_diff(data = test2_2_sub, user_id = test2_id)
X_test2['level_diff'] = test2_level_diff

index_list = []
for i in range(0, len(train3)):
    if(train3['acc_id'][i] in train_id):
        index_list.append(i)            
train3_sub = train3.loc[index_list]

index_list = []
for i in range(0, len(train3)):
    if(train3['acc_id'][i] in test_id):
        index_list.append(i)            
test3_sub = train3.loc[index_list]

index_list = []
for i in range(0, len(test1_3)):
    if(test1_3['acc_id'][i] in test1_id):
        index_list.append(i)            
test1_3_sub = test1_3.loc[index_list]

index_list = []
for i in range(0, len(test2_3)):
    if(test2_3['acc_id'][i] in test2_id):
        index_list.append(i)            
test2_3_sub = test2_3.loc[index_list]

def char_num(data, variable, user_id, version = 1):
    '''
    유저 아이디 별로 원하는 변수에 대한 unique값 count
    원하는 user_id에 대한 sub data가 있는 경우 version = 1, 없는 경우 version = 2
    version = 2 의 경우 subset 과정으로 시간이 더 오래 소요
    '''
    
    if(version == 1):
        data_sub = data
    else:
        index_list = []
        for i in range(0, len(data)):
            if(data['acc_id'][i] in user_id):
                index_list.append(i)            
        data_sub = data.loc[index_list]
    
    variable_unique = data_sub.groupby('acc_id')[variable].unique()
    variable_num = np.zeros(len(user_id))

    for i in range(0, len(user_id)):
        variable_num[i] = len(variable_unique[user_id[i]])
        
    return variable_num

pledge_train_id = train3_sub['acc_id'].unique()
pledge_test_id = test3_sub['acc_id'].unique()
pledge_test1_id = test1_3_sub['acc_id'].unique()
pledge_test2_id = test2_3_sub['acc_id'].unique()

train_pledge_num = char_num(train3_sub, 'pledge_id', pledge_train_id)
train_pledge_num = np.concatenate([pledge_train_id.reshape(-1, 1), train_pledge_num.reshape(-1, 1)], axis = 1)
train_pledge_num = pd.DataFrame(train_pledge_num)
train_pledge_num.columns = ['acc_id', 'pledge_num']
X_train = pd.merge(X_train, train_pledge_num, how = 'left', on = 'acc_id')
X_train = X_train.fillna(0)

test_pledge_num = char_num(test3_sub, 'pledge_id', pledge_test_id)
test_pledge_num = np.concatenate([pledge_test_id.reshape(-1, 1), test_pledge_num.reshape(-1, 1)], axis = 1)
test_pledge_num = pd.DataFrame(test_pledge_num)
test_pledge_num.columns = ['acc_id', 'pledge_num']
X_test = pd.merge(X_test, test_pledge_num, how = 'left', on = 'acc_id')
X_test = X_test.fillna(0)

test1_pledge_num = char_num(test1_3_sub, 'pledge_id', pledge_test1_id)
test1_pledge_num = np.concatenate([pledge_test1_id.reshape(-1, 1), test1_pledge_num.reshape(-1, 1)], axis = 1)
test1_pledge_num = pd.DataFrame(test1_pledge_num)
test1_pledge_num.columns = ['acc_id', 'pledge_num']
X_test1 = pd.merge(X_test1, test1_pledge_num, how = 'left', on = 'acc_id')
X_test1 = X_test1.fillna(0)

test2_pledge_num = char_num(test2_3_sub, 'pledge_id', pledge_test2_id)
test2_pledge_num = np.concatenate([pledge_test2_id.reshape(-1, 1), test2_pledge_num.reshape(-1, 1)], axis = 1)
test2_pledge_num = pd.DataFrame(test2_pledge_num)
test2_pledge_num.columns = ['acc_id', 'pledge_num']
X_test2 = pd.merge(X_test2, test2_pledge_num, how = 'left', on = 'acc_id')
X_test2 = X_test2.fillna(0)

def first_day(data, variable, user_id, version = 1):
    '''
    유저 아이디 별로 원하는 변수에 대한 unique값 count
    원하는 user_id에 대한 sub data가 있는 경우 version = 1, 없는 경우 version = 2
    version = 2 의 경우 subset 과정으로 시간이 더 오래 소요
    '''
    
    if(version == 1):
        data_sub = data
    else:
        index_list = []
        for i in range(0, len(data)):
            if(data['acc_id'][i] in user_id):
                index_list.append(i)            
        data_sub = data.loc[index_list]
    
    variable_unique = data_sub.groupby('acc_id')[variable].unique()
    variable_num = np.zeros(len(user_id))

    for i in range(0, len(user_id)):
        variable_num[i] = variable_unique[user_id[i]][0]
        
    return variable_num

first_log_day = first_day(train1_sub, 'day', train_id, version = 1)
X_train['first_log'] = first_log_day

first_log_day = first_day(test1_sub, 'day', test_id, version = 1)
X_test['first_log'] = first_log_day

first_log_day = first_day(test1_1_sub, 'day', test1_id, version = 1)
X_test1['first_log'] = first_log_day

first_log_day = first_day(test2_1_sub, 'day', test2_id, version = 1)
X_test2['first_log'] = first_log_day

day1 = (train1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 0:7].sum(axis = 1)
day2 = (train1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 7:14].sum(axis = 1)
day3 = (train1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 14:21].sum(axis = 1)
day4 = (train1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 21:28].sum(axis = 1)

X_train['log_1'] = day1.values
X_train['log_2'] = day2.values
X_train['log_3'] = day3.values
X_train['log_4'] = day4.values

day1 = (test1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 0:7].sum(axis = 1)
day2 = (test1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 7:14].sum(axis = 1)
day3 = (test1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 14:21].sum(axis = 1)
day4 = (test1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 21:28].sum(axis = 1)

X_test['log_1'] = day1.values
X_test['log_2'] = day2.values
X_test['log_3'] = day3.values
X_test['log_4'] = day4.values

day1 = (test1_1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 0:7].sum(axis = 1)
day2 = (test1_1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 7:14].sum(axis = 1)
day3 = (test1_1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 14:21].sum(axis = 1)
day4 = (test1_1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 21:28].sum(axis = 1)

X_test1['log_1'] = day1.values
X_test1['log_2'] = day2.values
X_test1['log_3'] = day3.values
X_test1['log_4'] = day4.values

day1 = (test2_1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 0:7].sum(axis = 1)
day2 = (test2_1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 7:14].sum(axis = 1)
day3 = (test2_1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 14:21].sum(axis = 1)
day4 = (test2_1_sub.groupby(['acc_id', 'day'])['playtime'].sum().unstack('day').fillna(0) != 0).iloc[:, 21:28].sum(axis = 1)

X_test2['log_1'] = day1.values
X_test2['log_2'] = day2.values
X_test2['log_3'] = day3.values
X_test2['log_4'] = day4.values

week1 = (train3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 0:7].sum(axis =1)
week2 = (train3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 7:14].sum(axis =1)
week3 = (train3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 14:21].sum(axis =1)
week4 = (train3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 21:28].sum(axis =1)

week1 = pd.DataFrame(week1)
week1.columns = ['pledge_log_1']
week2 = pd.DataFrame(week2)
week2.columns = ['pledge_log_2']
week3 = pd.DataFrame(week3)
week3.columns = ['pledge_log_3']
week4 = pd.DataFrame(week4)
week4.columns = ['pledge_log_4']

X_train = pd.merge(X_train, week1, how = 'left', on = 'acc_id')
X_train = pd.merge(X_train, week2, how = 'left', on = 'acc_id')
X_train = pd.merge(X_train, week3, how = 'left', on = 'acc_id')
X_train = pd.merge(X_train, week4, how = 'left', on = 'acc_id')
X_train = X_train.fillna(0)
X_train['pledge_log_total'] = X_train['pledge_log_1'] +  X_train['pledge_log_2'] + X_train['pledge_log_3'] + X_train['pledge_log_4']

week1 = (test3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 0:7].sum(axis =1)
week2 = (test3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 7:14].sum(axis =1)
week3 = (test3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 14:21].sum(axis =1)
week4 = (test3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 21:28].sum(axis =1)

week1 = pd.DataFrame(week1)
week1.columns = ['pledge_log_1']
week2 = pd.DataFrame(week2)
week2.columns = ['pledge_log_2']
week3 = pd.DataFrame(week3)
week3.columns = ['pledge_log_3']
week4 = pd.DataFrame(week4)
week4.columns = ['pledge_log_4']

X_test = pd.merge(X_test, week1, how = 'left', on = 'acc_id')
X_test = pd.merge(X_test, week2, how = 'left', on = 'acc_id')
X_test = pd.merge(X_test, week3, how = 'left', on = 'acc_id')
X_test = pd.merge(X_test, week4, how = 'left', on = 'acc_id')
X_test = X_test.fillna(0)
X_test['pledge_log_total'] = X_test['pledge_log_1'] +  X_test['pledge_log_2'] + X_test['pledge_log_3'] + X_test['pledge_log_4']

week1 = (test1_3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 0:7].sum(axis =1)
week2 = (test1_3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 7:14].sum(axis =1)
week3 = (test1_3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 14:21].sum(axis =1)
week4 = (test1_3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 21:28].sum(axis =1)

week1 = pd.DataFrame(week1)
week1.columns = ['pledge_log_1']
week2 = pd.DataFrame(week2)
week2.columns = ['pledge_log_2']
week3 = pd.DataFrame(week3)
week3.columns = ['pledge_log_3']
week4 = pd.DataFrame(week4)
week4.columns = ['pledge_log_4']

X_test1 = pd.merge(X_test1, week1, how = 'left', on = 'acc_id')
X_test1 = pd.merge(X_test1, week2, how = 'left', on = 'acc_id')
X_test1 = pd.merge(X_test1, week3, how = 'left', on = 'acc_id')
X_test1 = pd.merge(X_test1, week4, how = 'left', on = 'acc_id')
X_test1 = X_test1.fillna(0)
X_test1['pledge_log_total'] = X_test1['pledge_log_1'] +  X_test1['pledge_log_2'] + X_test1['pledge_log_3'] + X_test1['pledge_log_4']

week1 = (test2_3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 0:7].sum(axis =1)
week2 = (test2_3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 7:14].sum(axis =1)
week3 = (test2_3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 14:21].sum(axis =1)
week4 = (test2_3_sub.groupby(['acc_id', 'day'])['play_char_cnt'].sum().unstack('day').fillna(0) != 0).iloc[:, 21:28].sum(axis =1)

week1 = pd.DataFrame(week1)
week1.columns = ['pledge_log_1']
week2 = pd.DataFrame(week2)
week2.columns = ['pledge_log_2']
week3 = pd.DataFrame(week3)
week3.columns = ['pledge_log_3']
week4 = pd.DataFrame(week4)
week4.columns = ['pledge_log_4']

X_test2 = pd.merge(X_test2, week1, how = 'left', on = 'acc_id')
X_test2 = pd.merge(X_test2, week2, how = 'left', on = 'acc_id')
X_test2 = pd.merge(X_test2, week3, how = 'left', on = 'acc_id')
X_test2 = pd.merge(X_test2, week4, how = 'left', on = 'acc_id')
X_test2 = X_test2.fillna(0)
X_test2['pledge_log_total'] = X_test2['pledge_log_1'] +  X_test2['pledge_log_2'] + X_test2['pledge_log_3'] + X_test2['pledge_log_4']

X_train.to_csv('./train_preprocess_2.csv', index = False)
X_test.to_csv('./test_preprocess_2.csv', index = False)
X_test1.to_csv('./test1_preprocess_2.csv', index = False)
X_test2.to_csv('./test2_preprocess_2.csv', index = False)