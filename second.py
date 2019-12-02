import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import gc
import pickle
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingRegressor,StackingCVRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn import metrics
save_path = './data_final/'
def fun(x):
    if x >= 0.5:
        return 1
    else:
        return 0
with open(save_path + 'invite_info.pkl','rb') as file:
    invite_info = pickle.load(file)
member_feat = pd.read_hdf(save_path + 'member_feat.h5',key='data')
question_feat = pd.read_hdf(save_path + 'question_feat.h5',key='data')
member_question_feat = pd.read_hdf(save_path + 'member_question_feat.h5',key='data')
invite_info_evaluate = invite_info.ix[:1000]
invite_info_test = invite_info.ix[1000:2000]
invite_info = invite_info.ix[2000:]
tt = invite_info_evaluate['label']
ttt = invite_info_test['label']
del invite_info_evaluate['label'],invite_info_test['label']
invite_info['author_question_id'] = invite_info['author_id'] + invite_info['question_id']
invite_info_evaluate['author_question_id'] = invite_info_evaluate['author_id'] + invite_info_evaluate['question_id']
invite_info_test['author_question_id'] = invite_info_test['author_id'] + invite_info_test['question_id']
train = invite_info.merge(member_feat, 'left', 'author_id')
test = invite_info_evaluate.merge(member_feat, 'left', 'author_id')
pre = invite_info_test.merge(member_feat,'left','author_id')
train = train.merge(question_feat, 'left', 'question_id')
test = test.merge(question_feat, 'left', 'question_id')
pre = pre.merge(question_feat,'left','question_id')
train = train.merge(member_question_feat, 'left', 'author_question_id')
test = test.merge(member_question_feat, 'left', 'author_question_id')
pre = pre.merge(member_question_feat,'left','author_question_id')
del member_feat, question_feat, member_question_feat
gc.collect()
drop_feats = ['question_id', 'author_id', 'author_question_id', 'invite_time', 'label', 'invite_day']
used_feats = [f for f in train.columns if f not in drop_feats]
train_x = train[used_feats].reset_index(drop=True)
train_y = train['label'].reset_index(drop=True)
test_x = test[used_feats].reset_index(drop=True)
pre_x = pre[used_feats].reset_index(drop=True)
# LGBMClassifier
'''
model_lgb = LGBMClassifier(boostiong_type='gdbt',num_leaves=64,learning_rate=0.01,n_estimators=2500,max_bin=425,subsample_for_bin=50000,objective='binary',min_split_gain=0,min_child_weight=5,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=1,req_alpha=3,reg_lambda=5,seed=1000,n_jobs=-1,silent=True)
model_lgb.fit(train_x,train_y,eval_names=['train'],eval_metric=['logloss','auc'],eval_set=[(train_x,train_y)],early_stopping_rounds=10)
test_y = model_lgb.predict_proba(test_x)[:,1]
print("test auc: ",metrics.roc_auc_score(tt,test_y))
test_append = invite_info_evaluate[['question_id', 'author_id', 'invite_time']]
test_append['answer'] = test_y
test_append['answer'] = test_append['answer'].apply(lambda x: fun(x))
test_append['true'] = tt
test_append.to_csv('result_test_LGBM.txt',index=False,header=False,sep='\t')
pre_y = model_lgb.predict_proba(pre_x)[:,1]
print("pre auc: ",metrics.roc_auc_score(ttt,pre_y))
pre_append = invite_info_test[['question_id', 'author_id', 'invite_time']]
pre_append['answer'] = pre_y
pre_append['answer'] = pre_append['answer'].apply(lambda x: fun(x))
pre_append['true'] = ttt
pre_append.to_csv('result_pre_LGBM.txt',index=False,header=False,sep='\t')
'''
'''
# Stacking One
print("Let's Begin Stacking Model One")
lr = LinearRegression()
ridge = Ridge(random_state = 2019,)
models = [lr,ridge]
for model in models:
    model.fit(train_x,train_y)
    pred = model.predict(test_x)
    print("loss is {}".format(mean_squared_error(tt,pred)))
sclf = StackingRegressor(regressors = models,meta_regressor = ridge)
sclf.fit(train_x,train_y)
test_y= sclf.predict(test_x)
print("test auc: ",metrics.roc_auc_score(tt,test_y))
test_append = invite_info_evaluate[['question_id', 'author_id', 'invite_time']]
test_append['answer'] = test_y
test_append['answer'] = test_append['answer'].apply(lambda x: fun(x))
test_append['true'] = tt
test_append.to_csv('result_test_S1.txt',index=False,header=False,sep='\t')
pre_y= sclf.predict(pre_x)
print("pre auc: ",metrics.roc_auc_score(ttt,pre_y))
pre_append = invite_info_test[['question_id', 'author_id', 'invite_time']]
pre_append['answer'] = pre_y
pre_append['answer'] = pre_append['answer'].apply(lambda x: fun(x))
pre_append['true'] = ttt
pre_append.to_csv('result_pre_S1.txt',index=False,header=False,sep='\t')
'''
# Xgboost
'''
params = {'n_estimators': 10, 'seed': 0, 'n_estimators': 600, 'max_depth': 6, 'min_child_weight' :1, 'gamma': 0.2, 'subsample': 0.8, 'colsample_bytree': 0.9, 'reg_alpha':0.1, 'reg_lambda':0.05, 'learning_rate':0.01}
model = xgb.XGBRegressor(params=params, booster='dart')
model.fit(train_x, train_y)
test_y = model.predict(test_x)
print("test auc: ",metrics.roc_auc_score(tt,test_y))
test_append = invite_info_evaluate[['question_id', 'author_id', 'invite_time']]
test_append['answer'] = test_y
test_append['answer'] = test_append['answer'].apply(lambda x: fun(x))
test_append['true'] = tt
test_append.to_csv('result_test_Xgboost.txt',index=False,header=False,sep='\t')
pre_y= model.predict(pre_x)
print("pre auc: ",metrics.roc_auc_score(ttt,pre_y))
pre_append = invite_info_test[['question_id', 'author_id', 'invite_time']]
pre_append['answer'] = pre_y
pre_append['answer'] = pre_append['answer'].apply(lambda x: fun(x))
pre_append['true'] = ttt
pre_append.to_csv('result_pre_Xgboost.txt',index=False,header=False,sep='\t')
'''
# Stacking_Two

print("Let's Begin Stacking Model Two!")
lr = LinearRegression()
ridge = Ridge(random_state=2019,)
lasso =Lasso()
models = [lr,ridge, lasso]
params = {'lasso__alpha': [0.1, 1.0, 10.0],'ridge__alpha': [0.1, 1.0, 10.0]}
sclf = StackingCVRegressor(regressors=models, meta_regressor=ridge)
sclf = StackingCVRegressor(regressors=models, meta_regressor=ridge)
grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True)
grid.fit(train_x, train_y)
test_y = grid.predict(test_x)
print("test auc: ",metrics.roc_auc_score(tt,test_y))
test_append = invite_info_evaluate[['question_id', 'author_id', 'invite_time']]
test_append['answer'] = test_y
test_append['answer'] = test_append['answer'].apply(lambda x: fun(x))
test_append['true'] = tt
test_append.to_csv('result_test_S2.txt',index=False,header=False,sep='\t')
pre_y= grid.predict(pre_x)
print("pre auc: ",metrics.roc_auc_score(ttt,pre_y))
pre_append = invite_info_test[['question_id', 'author_id', 'invite_time']]
pre_append['answer'] = pre_y
pre_append['answer'] = pre_append['answer'].apply(lambda x: fun(x))
pre_append['true'] = ttt
pre_append.to_csv('result_pre_S2.txt',index=False,header=False,sep='\t')

