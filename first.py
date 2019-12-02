import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import multiprocessing as mp
import pickle
import gc
import os
from sklearn.preprocessing import LabelEncoder

def parse_a(d):
    return np.array(list(map(float, d.split())))

def parse_b(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))

def parse_c(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))
            
def parse_d(d):
    if d == '-1':
        return {}    
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))
path = './data/'
save_path = './data_final/'
if not os.path.exists(save_path):
    print("Create save_path~")
    os.mkdir(save_path)

invite_info = pd.read_csv(os.path.join(path, 'data.txt'), names=['question_id', 'author_id', 'invite_time', 'label'], sep='\t')
invite_info['invite_day'] = invite_info['invite_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
invite_info['invite_hour'] = invite_info['invite_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)

with open(save_path+'invite_info.pkl', 'wb') as file:
    pickle.dump(invite_info, file)

member_info = pd.read_csv(os.path.join(path, 'user_info.txt'), names=['author_id', 'gender', 'keyword', 'grade', 'hotness', 'reg_type','reg_plat','freq','A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2','score', 'topic_attent', 'topic_interest'], sep='\t')
member_info['topic_attent'] = member_info['topic_attent'].apply(parse_b)
member_info['topic_interest'] = member_info['topic_interest'].apply(parse_d)
# 扔掉第六个和第七个参数reg_type和reg_plat
del member_info["reg_type"],member_info["reg_plat"]
member_cat_feats = ['gender', 'freq','A2', 'B2', 'C2', 'D2', 'E2']
for feat in member_cat_feats:
    member_info[feat] = LabelEncoder().fit_transform(member_info[feat])

member_info['num_atten_topic'] = member_info['topic_attent'].apply(len)
member_info['num_interest_topic'] = member_info['topic_interest'].apply(len)

def most_interest_topic(d):
    if len(d) == 0:
        return -1
    return list(d.keys())[np.argmax(list(d.values()))]

def get_interest_values(d):
    if len(d) == 0:
        return [0]
    return list(d.values())

member_info['most_interest_topic'] = member_info['topic_interest'].apply(most_interest_topic)
member_info['most_interest_topic'] = LabelEncoder().fit_transform(member_info['most_interest_topic'])
member_info['interest_values'] = member_info['topic_interest'].apply(get_interest_values)
member_info['min_interest_values'] = member_info['interest_values'].apply(np.min)
member_info['max_interest_values'] = member_info['interest_values'].apply(np.max)
member_info['mean_interest_values'] = member_info['interest_values'].apply(np.mean)
member_info['std_interest_values'] = member_info['interest_values'].apply(np.std)

feats = ['author_id', 'gender', 'freq', 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2', 'score']
feats += ['num_atten_topic', 'num_interest_topic', 'most_interest_topic']
feats += ['min_interest_values', 'max_interest_values', 'mean_interest_values', 'std_interest_values']

for feat in [ 'gender', 'freq', 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2']:
    col_name = '{}_count'.format(feat)
    member_info[col_name] = member_info[feat].map(member_info[feat].value_counts().astype(int))
    member_info.loc[member_info[col_name] < 2 , feat] = -1
    member_info[feat] += 1
    member_info[col_name] = member_info[feat].map(member_info[feat].value_counts().astype(int))
    member_info[col_name] = (member_info[col_name] - member_info[col_name].min()) / (member_info[col_name].max() - member_info[col_name].min())
    feats += [col_name]
member_feat = member_info[feats]
member_feat.to_hdf(save_path+'member_feat.h5', key='data')

question_info = pd.read_csv(os.path.join(path, 'ques_info.txt'),names=['question_id', 'question_time', 'title_sw_series', 'title_w_series', 'desc_sw_series', 'desc_w_series', 'topic'], sep='\t')
question_info['title_sw_series'] = question_info['title_sw_series'].apply(parse_c)
question_info['title_w_series'] = question_info['title_w_series'].apply(parse_b)
question_info['desc_sw_series'] = question_info['desc_sw_series'].apply(parse_c)
question_info['desc_w_series'] = question_info['desc_w_series'].apply(parse_b)
question_info['topic'] = question_info['topic'].apply(parse_b)
question_info['question_day'] = question_info['question_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
question_info['question_hour'] = question_info['question_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)

question_info['num_title_sw'] = question_info['title_sw_series'].apply(len)
question_info['num_title_w'] = question_info['title_w_series'].apply(len)
question_info['num_desc_sw'] = question_info['desc_sw_series'].apply(len)
question_info['num_desc_w'] = question_info['desc_w_series'].apply(len)
question_info['num_qtopic'] = question_info['topic'].apply(len)
feats = ['question_id', 'num_title_sw', 'num_title_w', 'num_desc_sw', 'num_desc_w', 'num_qtopic', 'question_day','question_hour']
question_feat = question_info[feats]
question_feat.to_hdf(save_path+'question_feat.h5', key='data')

invite = invite_info
invite_id = invite[['author_id', 'question_id']]
invite_id['author_question_id'] = invite_id['author_id'] + invite_id['question_id']
invite_id.drop_duplicates(subset='author_question_id',inplace=True)
invite_id_qm = invite_id.merge(member_info[['author_id', 'topic_attent', 'topic_interest']], 'left', 'author_id').merge(question_info[['question_id', 'topic']], 'left', 'question_id')

def split_df(df, n):
    chunk_size = int(np.ceil(len(df) / n))
    return [df[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
def gc_mp(pool, ret, chunk_list):
    del pool
    for r in ret:
        del r
    del ret
    for cl in chunk_list:
        del cl
    del chunk_list
    gc.collect()

def process(df):
    return df.apply(lambda row: list(set(row['topic_attent']) & set(row['topic'])),axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_attent_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

def process1(df):
    return df.apply(lambda row: list(set(row['topic_interest'].keys()) & set(row['topic'])),axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process1, chunk_list)
invite_id_qm['topic_interest_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

def process2(df):
    return df.apply(lambda row: [row['topic_interest'][t] for t in row['topic_interest_intersection']],axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process2, chunk_list)
invite_id_qm['topic_interest_intersection_values'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

invite_id_qm['num_topic_attent_intersection'] = invite_id_qm['topic_attent_intersection'].apply(len)
invite_id_qm['num_topic_interest_intersection'] = invite_id_qm['topic_interest_intersection'].apply(len)

invite_id_qm['topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(lambda x: [0] if len(x) == 0 else x)
invite_id_qm['min_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.min)
invite_id_qm['max_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.max)
invite_id_qm['mean_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.mean)
invite_id_qm['std_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.std)
feats = ['author_question_id', 'num_topic_attent_intersection', 'num_topic_interest_intersection', 'min_topic_interest_intersection_values', 'max_topic_interest_intersection_values', 'mean_topic_interest_intersection_values', 'std_topic_interest_intersection_values']
feats += []
member_question_feat = invite_id_qm[feats]
member_question_feat.to_hdf(save_path+'member_question_feat.h5', key='data')
del invite_id_qm, member_question_feat,question_info, question_feat,member_feat, member_info
gc.collect()
