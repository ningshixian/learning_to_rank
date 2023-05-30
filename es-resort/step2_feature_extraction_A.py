# -*- coding: utf-8 -*-

import sys
import subprocess
import re
import math
import pickle
import time

import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import codecs
from tqdm import tqdm
from textdistance import cosine, jaccard
from gensim import corpora,similarities,models
from textdistance import cosine, jaccard, Hamming
import setting
import helpers

sys.path.append(r"../../")
import utils_toolkit as ut
import yg_jieba as jieba

"""
answer特征待定
特征抽取
构造 libsvm 训练数据集
划分训练/验证/测试集
"""

# 读取配置文件
DB_CONFIG, SBERT_CONFIG = helpers.get_config_info()
stpwrdlst = ut.StringHelper.get_stopword(setting.stopword_file)  # 停用词

# 全局变量
max_page_size = 50
feats_need_list = [ 'field_match_ratio',
    'num_jaccard_sim','num_edict_jaro','num_edict_ratio','num_edict_jaro_winkler',
    "sbert_cos_sim", "time_diff"
]
# feats_need_list = ["ctr", "clicks", "views", "num_key_in_query", "num_key_in_query_rate", 
#     'num_common_words', 'num_common_words_ratio1', 'num_common_words_ratio2',
#     'num_jaccard_sim','num_edict_distance_k_pt', 'num_edict_jaro','num_edict_ratio','num_edict_jaro_winkler',
#     'tfidf_cos_sim', "sbert_cos_sim", "string_cosine", "Hamming_kt", "Hamming_sim_kt",
#     "time_diff","searchOrder"
# ]
with codecs.open(setting.data_bucketA + setting.feature_all_file, 'w', 'utf-8') as f:
    f.write('\n'.join(feats_need_list))
nb_categorical_feature = -0  # 取消特征缩放
print("倒数{}个特征是类别特征".format(nb_categorical_feature))  # -3

# 读取数据
data = pd.read_csv(setting.data_bucketA + setting.data_csv)
data = data.fillna('0')    # '0'
# data = dataframe.copy()


# 关键词在query中出现的次数/比率
if 'num_key_in_query' in feats_need_list:
    
    #描述在key_word中出现的次数
    def get_num_key(x,y):
        if str(y)=='0' or str(y)=='':
            return -1
        y=y.split('###')
        num=0
        for kw in y:
            if kw in x:
                num+=1
        return num

    data['num_key_in_query']=list(map(lambda x,y: get_num_key(x,y), data['input'],data['key_word']))
    data['num_key_in_query_rate']=list(map(lambda x,y: 0 if x==-1 else x/len(y.split('###')),
                                                data['num_key_in_query'], data['key_word']))


# 统计quey-doc的词共现
if "num_common_words" in feats_need_list:
    
    def get_num_common_words_and_ratio(merge, col):
        merge = merge[col]
        merge.columns = ['q1', 'q2']
        merge['q2'] = merge['q2'].apply(lambda x: '0' if str(x) == 'nan' else x)
        q1_word_set = merge.q1.apply(set).values
        q2_word_set = merge.q2.apply(set).values
        q1_word_len_set = merge.q1.apply(lambda x: len(set(x))).values
        q2_word_len_set = merge.q2.apply(lambda x: len(set(x))).values
        result = [len(q1_word_set[i] & q2_word_set[i]) for i in range(len(q1_word_set))]
        result_ratio_q_set = [result[i] / q1_word_len_set[i] for i in range(len(q1_word_set))]
        result_ratio_t_set = [result[i] / q2_word_len_set[i] for i in range(len(q1_word_set))]
        return result, result_ratio_q_set, result_ratio_t_set

    data['num_common_words'], \
    data['num_common_words_ratio1'], \
    data['num_common_words_ratio2'] = get_num_common_words_and_ratio(data, ['input', 'answer'])  # primaryQues


# 场景1：查询词和召回字段匹配程度排序优化
# 解决方案：应用排序相关性特征函数项，获取某字段上与查询词匹配的分词词组个数与该字段总词组个数的比值
# https://help.aliyun.com/document_detail/193400.html?spm=5176.11065259.1996646101.searchclickresult.1cfd1ebcoRSjKJ
if "field_match_ratio" in feats_need_list:
    
    def get_field_match_ratio(merge, col):
        merge = merge[col]
        merge.columns = ['q1', 'q2']
        merge['q2'] = merge['q2'].apply(lambda x: '0' if str(x) == 'nan' else x)
        q1_word_set = merge.q1.apply(lambda x: set(jieba.lcut(x))).values
        q2_word_set = merge.q2.apply(lambda x: set(jieba.lcut(x))).values
        result = [len(q1_word_set[i] & q2_word_set[i]) for i in range(len(q1_word_set))]
        result_ratio_t_set = [result[i] / len(q2_word_set[i]) for i in range(len(q1_word_set))]
        return result_ratio_t_set

    data['field_match_ratio'] = get_field_match_ratio(data, ['input', 'primaryQues'])


# Jaccard 相似度
if "num_jaccard_sim" in feats_need_list:
    
    def jaccard(x, y):
        if str(y)=='':
            y = '0'
        x = set(x)
        y = set(y)
        return float(len(x & y) / len(x | y))

    data['num_jaccard_sim'] = list(
        map(lambda x, y: jaccard(x, y), data['input'], data['primaryQues']))


# 编辑距离
import Levenshtein
print('get edict distance:')
data['num_edict_distance_k_pt'] = list(
    map(lambda x, y: Levenshtein.distance(x, y) / (len(x)+1), tqdm(data['input']), data['primaryQues']))
data['num_edict_jaro'] = list(
    map(lambda x, y: Levenshtein.jaro(x, y), tqdm(data['input']), data['primaryQues']))
data['num_edict_ratio'] = list(
    map(lambda x, y: Levenshtein.ratio(x, y), tqdm(data['input']), data['primaryQues']))
data['num_edict_jaro_winkler'] = list(
    map(lambda x, y: Levenshtein.jaro_winkler(x, y), tqdm(data['input']), data['primaryQues']))


# 原始生预料
# sen_list = list(map(lambda x: 'q_'+x, data['input'].values)) + list(map(lambda x: 'a_'+x, data['primaryQues'].values))
sen_list = list(data['input'].values) + list(data['primaryQues'].values) + list(data['answer'].values)
sen_list = list(set(sen_list))


# tfidf余弦相似度
# MemoryError: Unable to allocate 8.57 GiB for an array with shape (143850, 8000) and data type float64
if "tfidf_cos_sim" in feats_need_list:
    
    def get_sen2tfidf(sen_list):
        print("tfidf句向量获取....")
        sen2tfidf = {}
        raw_documents = [" ".join(jieba.lcut(x)) for x in sen_list]  # 已分词
        tfidf_vectorizer = ut.VSM.tfidf(raw_documents, stpwrdlst)
        tfidf_vec_array = tfidf_vectorizer.transform(raw_documents).toarray()   # 矩阵过大
        # assert len(tfidf_vec_array) == len(sen_list)
        for i in range(len(sen_list)):
            sen = sen_list[i]
            sen2tfidf[sen] = tfidf_vec_array[i]
        # 保存必要的中间变量(用于预测)
        joblib.dump(tfidf_vectorizer, setting.data_bucketA + setting.tfidf_vec_file)
        return sen2tfidf

    sen2tfidf = get_sen2tfidf(sen_list)
    data['tfidf_cos_sim'] = list(map(lambda x, y: cosine_similarity([sen2tfidf[x]], [sen2tfidf[y]])[0][0], tqdm(data['input']), data['primaryQues']))
    sen2tfidf = None


# '词向量平均的相似度'
# from gensim.models import Word2Vec
# w2v_model = Word2Vec(corpus, size=300, window=8, min_count=0, workers=20, sg=1,iter=9)
# w2v_model.save('pretrain_model/w2v_300.model')
# w2v_model = Word2Vec.load('pretrain_model/w2v_300.model')


def get_sen2bertvec(sen_list):
    print("sbert句向量获取....")
    sen2bertvec, _ = helpers.get_sbertvec(sen_list, SBERT_CONFIG["url"])
    assert set(sen_list).issubset(list(sen2bertvec.keys()))
    return sen2bertvec


# sbert余弦相似度
sen2bertvec = {}
if "sbert_cos_sim" in feats_need_list:
    sen2bertvec = get_sen2bertvec(sen_list)
    data['sbert_cos_sim'] = list(map(lambda x, y: cosine_similarity([sen2bertvec[x]], [sen2bertvec[y]])[0][0], tqdm(data['input']), data['primaryQues']))
# if "answer_sbert_cos_sim" in feats_need_list:
#     if not sen2bertvec:
#         sen2bertvec = get_sen2bertvec(sen_list)
#     data['answer_sbert_cos_sim'] = list(map(lambda x, y: cosine_similarity([sen2bertvec[x]], [sen2bertvec[y]])[0][0], tqdm(data['input']), data['answer']))
sen2bertvec = None


data['string_cosine'] = list(
    map(lambda x, y: cosine.similarity(x, y), tqdm(data['input']), data['primaryQues']))
data['Hamming_kt'] = list(map(lambda x, y: Hamming(qval=None).normalized_distance(x, y),
                                        tqdm(data['input']), data['primaryQues']))
data['Hamming_sim_kt'] = list(map(lambda x, y: 
                                        Hamming(qval=None).similarity(x, y),
                                        tqdm(data['input']), data['primaryQues']))


print("构造 libsvm 格式的数据")
qid, prev = 0, ""
duplicate_items = set() # 避免重复条目
with codecs.open(setting.data_bucketA + setting.ltr_data_txt, "w", "utf-8") as f:
    for index, row in data.iterrows():
        query = row["input"]
        label = row["label"]
        
        # 控制qid递增
        if query != prev: 
            qid += 1
            prev = query
            duplicate_items = set()  # 新的qid重置
        
        # 一行特征值构造
        line = "{} qid:{} ".format(label, qid)
        feats = [row[name] for name in feats_need_list]  # 只写入feats_need_list中包含的特征
        line += " ".join(["{}:{}".format(idx+1, round(float(feat), 5)) for idx, feat in enumerate(feats)])
        line += " #{} {} {}".format(
            repr(re.sub(" ", "", query)),
            repr(re.sub(" ", "", row["primaryQues"])),
            repr(re.sub(" ", "", row["botCode"])),
        )
        
        # 避免重复条目写入文件
        if line.split("#")[0] not in duplicate_items:
            duplicate_items.add(line.split("#")[0])
            f.write(line)
            f.write("\n")


duplicate_items = set()


print("划分训练/验证/测试集")
with codecs.open(setting.data_bucketA + setting.ltr_data_txt, "r", "utf-8") as f:
    lines = f.readlines()
totals = len(lines)
a = int(totals * 0.8)
b = int(totals * 0.9)
with codecs.open(setting.data_bucketA + setting.ltr_train, "w", "utf-8") as f:
    for i in range(0, b):
        f.write(lines[i])
with codecs.open(setting.data_bucketA + setting.ltr_valid, "w", "utf-8") as f:
    for i in range(b, totals):
        f.write(lines[i])
with codecs.open(setting.data_bucketA + setting.ltr_test, "w", "utf-8") as f:
    for i in range(b, totals):
        f.write(lines[i])

# # lambdarank task, split according to groups
# # GroupKFold 会保证同一个group的数据不会同时出现在训练集和测试集上。
# # 因为如果训练集中包含了每个group的几个样例，可能训练得到的模型能够足够灵活地从这些样例中学习到特征，在测试集上也会表现很好。
# # 但一旦遇到一个新的group它就会表现很差。
# # 参考：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
# from sklearn.model_selection import GroupKFold
# group_kfold = GroupKFold(n_splits=2)
# group_kfold.get_n_splits(x_train, y_train, q_train)
# for train_index, test_index in group_kfold.split(x_train, y_train, q_train):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     # X_train, X_test = x_train[train_index], x_train[test_index]
#     # y_train, y_test = y_train[train_index], y_train[test_index]
#     # print(X_train, X_test, y_train, y_test)

print("划分特征集和group集")
loader = subprocess.Popen(
    [
        "python",
        "trans_data.py",
        setting.data_bucketA + setting.ltr_train,
        setting.data_bucketA + setting.lxh_train,
        setting.data_bucketA + setting.lxh_train_group,
    ]
)
returncode = loader.wait()  # 阻塞直至子进程完成
loader = subprocess.Popen(
    [
        "python",
        "trans_data.py",
        setting.data_bucketA + setting.ltr_valid,
        setting.data_bucketA + setting.lxh_valid,
        setting.data_bucketA + setting.lxh_valid_group,
    ]
)
returncode = loader.wait()  # 阻塞直至子进程完成
loader = subprocess.Popen(
    [
        "python",
        "trans_data.py",
        setting.data_bucketA + setting.ltr_test,
        setting.data_bucketA + setting.lxh_test,
        setting.data_bucketA + setting.lxh_test_group,
    ]
)
returncode = loader.wait()  # 阻塞直至子进程完成


# # 保存必要的中间变量(用于预测)
# joblib.dump(nb_categorical_feature, setting.cat_feat_file)  # 倒数几个特征是属于类别特征
# joblib.dump(stdsc, setting.stdsc_file)  # 数据放缩字典
# joblib.dump(label_encoders, setting.label_encoder_file)  # 保存特征转换的字典
