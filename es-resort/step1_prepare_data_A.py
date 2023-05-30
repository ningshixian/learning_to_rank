# -*- coding: utf-8 -*-
import os
import math
import re
import sys
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
tqdm.pandas(desc='apply')
import codecs
from gensim.models.doc2vec import Doc2Vec
from textdistance import cosine, jaccard

import setting
import helpers

sys.path.append(r"../../")
import utils_toolkit as ut
import yg_jieba as jieba  # 未使用！

"""ES日志搜集和分析，构造训练数据
1、查询数据库，获取ES日志
2、字典存储query-features对
3、统计点击率
4、写入Excel
"""

# 可一键添加新特征
# col_name_output = [
#     "primaryQues,knowledgeId,baseCode,botCode,channel,projectCode,answer,searchOrder",
#     "primaryQuestion,knowledgeId,selectBaseCode,null 'botCode','0' as 'channel','0' as 'projectCode','0' as 'answer',searchOrder",
# ]
col_name_output = [
    "primaryQues,knowledgeId,baseCode,botCode,searchOrder",
    "primaryQuestion,knowledgeId,selectBaseCode,null as 'botCode',searchOrder",
]


def get_click_ratio(output_dataframe, data_es_click):
    """
    统计点击数Clicks、曝光数Impressions、点击率CTR
    PS: input和primaryQues需提前进行数据清洗！！
    """
    print("click清洗数据（耗时操作）...")
    click_dataframe = pd.DataFrame(data_es_click)    # 转换dataframe数据格式
    # click_dataframe.columns = ["input", "primaryQues"]
    # click_dataframe[['input', 'primaryQues']] = click_dataframe[['input', 'primaryQues']].progress_applymap(lambda x: helpers.clean(x, stpwrdlst))
    click_dataframe.columns = ["answer", "input", "primaryQues"]
    click_dataframe[['input', 'primaryQues']] = click_dataframe[['input', 'primaryQues']].progress_applymap(lambda x: helpers.clean(x, stpwrdlst))
    click_dataframe[['answer']] = click_dataframe[['answer']].progress_applymap(lambda x: helpers.clean_html(x))


    ctr, clicks, views = {}, {}, {}
    ctr_pair, clicks_pair, views_pair = {}, {}, {}  # 用于近似label

    # 统计点击数
    for index, row in click_dataframe.iterrows():
        if row['input']:
            # 针对 query-doc
            key = (row['input'], row['primaryQues'])
            clicks_pair.setdefault(key, 0)
            clicks_pair[key] += 1
            # 针对 doc
            clicks.setdefault(row['primaryQues'], 0)
            clicks[row['primaryQues']] += 1

    # 统计曝光数
    for index, row in output_dataframe.iterrows():
        if row['input']:
            # 针对 query-doc
            key = (row['input'], row['primaryQues'])
            views_pair.setdefault(key, 0)
            views_pair[key] += 1
            # 针对 doc
            views.setdefault(row['primaryQues'], 0)
            views[row['primaryQues']] += 1

    # 统计query-doc点击率
    for k, v in views_pair.items():
        score_pair = helpers.walson_ctr(clicks_pair.get(k, 0), views_pair.get(k, -1))
        if score_pair and not math.isnan(score_pair):  # float('nan')=not a number
            ctr_pair[k] = round(score_pair, 5)  # 点击率CTR修正-Wilson CTR
        else:
            ctr_pair[k] = 0.0
    # 统计doc点击率
    for k, v in views.items():
        score = helpers.walson_ctr(clicks.get(k, 0), views.get(k, -1))
        # 存在view>click的情况，导致sqrt错误
        if score and not math.isnan(score):  # float('nan')=not a number
            ctr[k] = round(score, 5)  # 点击率CTR修正-Wilson CTR
        else:
            ctr[k] = 0.0
    
    return ctr_pair, clicks_pair, views_pair


# 根据quey-doc点击率来近似label
def get_label(query, primary):
    if query and query==primary:    # 强制label=1
        label = "2"
    else:
        qd_ctr = ctr_pair.get((query, primary), 0)
        assert qd_ctr >= 0
        if qd_ctr > 1:
            print("点击率大于1", (query, primary), qd_ctr)
        label = "0" if qd_ctr == 0 else ("1" if qd_ctr > 0 and qd_ctr <= 0.4 else "2")
    return label


# 读取配置文件
DB_CONFIG, SBERT_CONFIG = helpers.get_config_info()
stpwrdlst = ut.StringHelper.get_stopword(setting.stopword_file)  # 停用词

print("执行SQL取es日志数据...")
helpers.mem_check()
mysql = ut.MySQLHelper.PooledDBConnection(DB_CONFIG)  # 数据库连接对象
sql_1 = """SELECT answer,input,{}
        FROM know_es_mobile_output
        WHERE yn = 1
            AND input IS NOT NULL
            AND input REGEXP '[^0-9.]' = 1
            AND TIMESTAMPDIFF(MONTH, opeTime,now()) <= 6""".format(col_name_output[0])    #  只取4个月以内的数据
data_es_output = mysql.ExecQuery(sql_1)
sql_2 = """SELECT answer, input, primaryQues
        FROM know_es_mobile_click
        WHERE yn = 1
            AND input IS NOT NULL
            AND input REGEXP '[^0-9.]' = 1
            AND TIMESTAMPDIFF(MONTH, opeTime,now()) <= 6"""
data_es_click = mysql.ExecQuery(sql_2)

# print("执行SQL取B端坐席的搜索日志...")
# sql_4 = """SELECT searchContent,{}
#         FROM know_seat_result
#         WHERE searchContent IS NOT NULL
#             AND searchContent REGEXP '[^0-9.]' = 1""".format(col_name_output[1])
# a = mysql.ExecQuery(sql_4)
# sql_5 = """SELECT searchContent, primaryQuestion
#         FROM know_seat_answer
#         WHERE searchContent IS NOT NULL
#             AND searchContent REGEXP '[^0-9.]' = 1"""
# c = mysql.ExecQuery(sql_5)
# data_es_output = np.append(data_es_output, a, axis=0)
# data_es_click = np.append(data_es_click, c, axis=0)

print("执行SQL取主问题知识...")
sql_3 = """SELECT knowledge_id, category_id, create_time, update_time, key_word
        FROM oc_knowledge_management
        WHERE yn = 1 
            AND `status` IN (1,2,4)
            AND base_code is not null
            AND base_code <> "XIANLIAOBASE"
            and base_code <> 'XHTXBASE'
            and base_code <> 'LXHITDH'
        """
data_es_cat = mysql.ExecQuery(sql_3)

# 时间差、类别、关键词
time_feats, cat_feats, kw_feats = {}, {}, {}
for item in data_es_cat:
    kid, cat_id, t1, t2, kw = str(item[0]), str(item[1]), item[2], item[3], item[4]
    if t2:  # 若时间更新
        diff = parse(str(t2)) - parse("2018-12-01 00:00:00")
    else:
        diff = parse(str(t1)) - parse("2018-12-01 00:00:00")
    time_feats[kid] = diff.days  # int
    cat_feats[kid] = cat_id
    kw_feats[kid] = kw.strip('###') if kw else ''


dataframe = pd.DataFrame(data_es_output)    # 转换dataframe数据格式
dataframe.columns = ['answer', 'input'] + col_name_output[0].split(',')    # # 修改列名
data_es_output = None
# dataframe = dataframe.drop_duplicates()  # 去重 → 会导致曝光数统计错误

print("过滤无意义数据....")
click_input_list = list(set([x[1] for x in data_es_click]))
dataframe = dataframe[dataframe['input'].str.len() > 1]  # 无意义单字Query过滤
dataframe = dataframe[~dataframe['input'].str.contains(r'[a-z]$')]  # 删掉以字母结尾的数据，添加~用于反转条件
dataframe = dataframe[dataframe['input'].isin(click_input_list)]   # 只保留有点击行为的数据
kid_list = list(map(lambda x: str(x), set(np.array(data_es_cat)[:, 0])))
dataframe = dataframe[dataframe['knowledgeId'].isin(kid_list)]  # 过滤不再使用的主问题噪音数据√
print("dataframe.shape: ", dataframe.shape)  # (1288346, 5)→只保留...→(345670, 5)

print("清洗数据（耗时操作）...")
# dataframe['answer'] = dataframe['answer'].apply(lambda x: re.sub("<[^>]*>", "", str(x)).replace('\n','').replace(' ',''))
# dataframe['answer'] = dataframe['answer'].progress_apply(lambda x: helpers.clean(str(x), stpwrdlst))
dataframe[['input', 'primaryQues']] = dataframe[['input', 'primaryQues']].progress_applymap(lambda x: helpers.clean(x, stpwrdlst))
dataframe[['answer']] = dataframe[['answer']].progress_applymap(lambda x: helpers.clean_html(x))
dataframe = dataframe[dataframe['input'].str.len() > 1]  # 无意义单字Query过滤

print("拓展特征维度...")
dataframe['key_word'] = dataframe['knowledgeId'].map(lambda x: kw_feats.get(str(x), ''))
dataframe['time_diff'] = dataframe['knowledgeId'].map(lambda x: time_feats.get(str(x), ''))
dataframe['category_id'] = dataframe['knowledgeId'].map(lambda x: cat_feats.get(str(x), ''))

print("获取点击统计特征...")
ctr_pair, clicks_pair, views_pair = get_click_ratio(dataframe, data_es_click)
dataframe['views'] = list(map(lambda x,y: views_pair.get((str(x), str(y)), 0), dataframe['input'], dataframe['primaryQues']))
dataframe['clicks'] = list(map(lambda x,y: clicks_pair.get((str(x), str(y)), 0), dataframe['input'], dataframe['primaryQues']))
dataframe['ctr'] = list(map(lambda x,y: ctr_pair.get((str(x), str(y)), 0), dataframe['input'], dataframe['primaryQues']))

print("获取对应的label...")
dataframe['label'] = list(map(lambda x,y: get_label(str(x), str(y)), dataframe['input'], dataframe['primaryQues'])) 
print("dataframe.shape: ", dataframe.shape)     # 

print("将DataFrame存储为csv")
dataframe = dataframe.drop_duplicates()  # 去重 → 会导致曝光数统计错误 → 该操作移动到写入csv文件前进行！
dataframe.to_csv(setting.data_bucketA + setting.data_csv, index=False, sep=',')
print("dataframe.shape: ", dataframe.shape)     #  (1288346, 12)

# 保存必要的中间变量(用于预测)
joblib.dump((ctr_pair, clicks_pair, views_pair), setting.data_bucketA + setting.ctr_file)
joblib.dump(time_feats, setting.data_bucketA + setting.time_feats_file)


# print("组装用于生成向量的生语料")
# sen_list = list(set(input2doc.keys())) + list(
#     set([item[0] for x in list(input2doc.values()) for item in x])
# )  # 未分词
# raw_documents = [" ".join(jieba.lcut(x)) for x in sen_list]  # 已分词

# sen2tfidf = {}
# if "score_tfidf" in header:
#     print("tfidf训练")
#     tfidf_vectorizer = ut.VSM.tfidf(raw_documents, stpwrdlst)
#     tfidf_vec_array = tfidf_vectorizer.transform(raw_documents).toarray()
#     assert len(tfidf_vec_array) == len(sen_list)
#     for i in range(len(sen_list)):
#         sen = sen_list[i]
#         sen2tfidf[sen] = tfidf_vec_array[i]
#     tfidf_vectorizer, tfidf_vec_array = None, None

# sen2bow = {}
# if "score_bow" in header:
#     print("bow训练")
#     bow_vectorizer = ut.VSM.bow(raw_documents, stpwrdlst)
#     bow_vec_array = bow_vectorizer.transform(raw_documents).toarray()
#     assert len(bow_vec_array) == len(sen_list)
#     for i in range(len(sen_list)):
#         sen = sen_list[i]
#         sen2bow[sen] = bow_vec_array[i]
#     bow_vec_array = []

# bm25 = None
# if "score_bm25" in header:
#     print("bm25训练")
#     bm25 = ut.VSM.BM25(stpwrdlst=stpwrdlst)
#     bm25.fit(raw_documents)

# sen2w2v = {}
# if "score_w2v" in header:
#     print("w2v-avg句向量")
#     embedding = ut.EmbeddingHelper.readTxtEmbedFile("", 200)
#     for s1 in raw_documents:
#         v1 = []
#         s1 = s1.split(" ")
#         for w in s1:
#             if w not in stpwrdlst and w in embedding:
#                 v1.append(embedding[w])
#             else:
#                 # 忽略OOV的情况
#                 v1.append(embedding["UNKNOWN_TOKEN"])
#         v1 = np.array(v1).mean(axis=0)
#         sen2w2v["".join(s1)] = v1
#     embedding = None

# sen2docvec = {}
# if "score_doc2vec" in header:
#     print("doc2vec训练......")
#     model = ut.VSM.doc2vec(raw_documents, stpwrdlst)
#     sen2docvec = {sen_list[i]: model.docvecs[i] for i in range(len(raw_documents))}

# sen2bertvec = {}
# if "score_bert" in header:
#     print("sbert句向量获取....")
#     sen2bertvec, _ = helpers.get_sbertvec(sen_list, SBERT_CONFIG["url"])
#     assert set(sen_list).issubset(list(sen2bertvec.keys()))


# helpers.mem_check()
# print("转换成pandas数据格式，并写入Excel...")
# res = []  # 用于组装一行数据
# for query, features in tqdm(input2doc.items()):
#     score_obj_dict = []
#     # 提前计算query-doc的余弦相似度
#     if sen2tfidf:
#         score_tfidf = cosine_similarity(
#             [sen2tfidf[query]], [sen2tfidf[p[0]] for p in features]
#         )
#         score_tfidf = score_tfidf.tolist()[0]
#         score_obj_dict.append(score_tfidf)
#     if sen2bow:
#         score_bow = cosine_similarity(
#             [sen2bow[query]], [sen2bow[p[0]] for p in features]
#         )
#         score_bow = score_bow.tolist()[0]
#         score_obj_dict.append(score_bow)
#     if sen2docvec:
#         score_doc2vec = cosine_similarity(
#             [sen2docvec[query]], [sen2docvec[p[0]] for p in features]
#         )
#         score_doc2vec = score_doc2vec.tolist()[0]
#         score_obj_dict.append(score_doc2vec)
#     if sen2bertvec:
#         score_bert = cosine_similarity(
#             [sen2bertvec[query]], [sen2bertvec[p[0]] for p in features]
#         )
#         score_bert = score_bert.tolist()[0]
#         score_obj_dict.append(score_bert)
#     if bm25:
#         docs = [" ".join(jieba.lcut(x[0])) for x in features]
#         score_bm25 = bm25.transform(" ".join(jieba.lcut(query)), docs).tolist()
#         score_obj_dict.append(score_bm25)

#     for i in range(len(features)):
#         feat = list(map(helpers.fillna, features[i]))  # col_name_output[0] + time
#         primary = feat[0]
#         kid = feat[1]
#         res.append([])  # 用于组装一行数据


# # 保存必要的中间变量(用于预测)
# if "ctr" in header:
#     joblib.dump((ctr, clicks, views), setting.ctr_file)
# if "score_tfidf" in header:
#     joblib.dump(tfidf_vectorizer, setting.tfidf_vec_file)
# if "score_bow" in header:
#     joblib.dump(bow_vectorizer, setting.bow_vec_file)
# if "score_bm25" in header:
#     joblib.dump(bm25, setting.bm25_file)
#     # joblib.dump(model, setting.doc2vec_file)
# if "score_doc2vec" in header:
#     model.save(setting.doc2vec_file)  # save dov2vec
#     # model.save_word2vec_format(setting.word2vec_file, binary=False) #save word2vec
#     joblib.dump(sen2docvec, setting.sen2docvec_dict_file)

# joblib.dump((time_feats, cat_feats), setting.time_feats_file)
