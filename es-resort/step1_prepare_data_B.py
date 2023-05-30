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
import codecs
from gensim.models.doc2vec import Doc2Vec
from textdistance import cosine, jaccard

import setting
import helpers

sys.path.append(r"../../")
import utils_toolkit as ut
import yg_jieba as jieba

"""ES日志搜集和分析，构造训练数据
1、查询数据库，获取ES日志
2、字典存储query-features对
3、统计点击率
4、写入Excel
"""

# 可一键添加新特征
col_name_output = [
    "primaryQues,knowledgeId,baseCode,botCode,searchOrder",
    "primaryQuestion,knowledgeId,selectBaseCode,null 'botCode',searchOrder",
]
# excel表头名称
header = (
    [
        "label",
        "query",
        # "ctr",（禁用）
        # "clicks",（禁用）
        # "views",（禁用）
        "score",  # 字符串形式的cosine相似度（必须）
        # "score_tfidf", "score_bow", "score_bm25", "score_doc2vec", "score_w2v"
        "score_bert",
    ]
    + col_name_output[0].split(",")
    + ["opeTime","time_diff", "category_id"]  # 
)
# 获取answer所在的索引
if "answer" in col_name_output[0]:
    answer_idx = col_name_output[0].split(",").index("answer")
else:
    answer_idx = None


def get_time_category(data_es_cat):
    """计算时间差和类别特征"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_feats, cat_feats = {}, {}
    for item in data_es_cat:
        kid, cat_id, t1, t2 = str(item[0]), str(item[1]), item[2], item[3]
        # diff = parse(t2) - parse(t1) if t2 else 0
        if t2:  # 若时间更新
            diff = parse(str(t2)) - parse("2018-12-01 00:00:00")
        else:
            diff = parse(str(t1)) - parse("2018-12-01 00:00:00")
        time_feats[kid] = diff.days  # int
        cat_feats[kid] = cat_id

    if "time_diff" not in header:
        time_feats = {}
    if "category_id" not in header:
        cat_feats = {}
    return time_feats, cat_feats


def store_data_in_dictionary(data_es_output):
    """将数据存储成字典格式 {query: feature_list}"""
    input2doc, tmp = {}, {}
    for i in range(len(data_es_output)):
        item = data_es_output[i]
        t = list(data_es_opeTime[i])
        inp = item[0].strip()
        features = list(item[1:])
        primary = features[0].strip()
        if answer_idx:  # 判断input是否被answer包含
            features[answer_idx] = (
                1 if inp in re.sub("<.*>", "", features[answer_idx]) else 0
            )
        if inp and primary:
            input2doc.setdefault(inp, [])
            tmp.setdefault(inp, [])
            if not features in tmp[inp]:  # 去重(不包括opeTime)
                tmp[inp].append(features)
                input2doc[inp].append(features + t)
    return input2doc


def get_click_ratio(data_es_output, data_es_click):
    """统计点击数Clicks、曝光数Impressions、点击率CTR
    PS:与用户搜索的query无关！？
    """
    ctr, clicks, views = {}, {}, {}
    ctr_pair, clicks_pair, views_pair = {}, {}, {}  # 用于近似label

    # 统计点击数
    for item in data_es_click:
        if item[0]:
            # 针对 query-doc
            key = (item[0], item[1])
            clicks_pair.setdefault(key, 0)
            clicks_pair[key] += 1
            # 针对 doc
            clicks.setdefault(item[1], 0)
            clicks[item[1]] += 1

    # 统计曝光数
    for item in data_es_output:
        if item[0]:
            # 针对 query-doc
            key = (item[0], item[1])
            views_pair.setdefault(key, 0)
            views_pair[key] += 1
            # 针对 doc
            views.setdefault(item[1], 0)
            views[item[1]] += 1

    # 统计点击率
    for k, v in views_pair.items():
        score_pair = helpers.walson_ctr(clicks_pair.get(k, 0), views_pair.get(k, -1))
        if score_pair and not math.isnan(score_pair):  # float('nan')=not a number
            ctr_pair[k] = round(score_pair, 5)  # 点击率CTR修正-Wilson CTR
        else:
            ctr_pair[k] = 0.0

    for k, v in views.items():
        score = helpers.walson_ctr(clicks.get(k, 0), views.get(k, -1))
        # 存在view>click的情况，导致sqrt错误
        if score and not math.isnan(score):  # float('nan')=not a number
            ctr[k] = round(score, 5)  # 点击率CTR修正-Wilson CTR
        else:
            ctr[k] = 0.0

    if "ctr" not in header:
        ctr, clicks, views = {}, {}, {}
    return ctr, clicks, views, ctr_pair


def clean_query_and_doc(q_list):
    # 只保留列表中每个句子的中文、数字、英文字母部分
    row = []
    for q in q_list:
        if q:
            new_q = helpers.clean(q, stpwrdlst)
            row.append(new_q)
        else:
            row.append("")
    return row


# 根据quey-doc点击率来近似label
def get_label(query, primary):
    if query and query==primary:    # 强制label=1
        label = "1"
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

print("执行SQL取数据...")
helpers.mem_check()
mysql = ut.MySQLHelper.PooledDBConnection(DB_CONFIG)  # 数据库连接对象
#  and opeTime LIKE '2020%'
sql_1 = """SELECT input,{}
        FROM know_es_mobile_output
        WHERE yn = 1
            AND input IS NOT NULL
            AND input REGEXP '[^0-9.]' = 1
            AND TIMESTAMPDIFF(MONTH, opeTime,now()) <= 6""".format(col_name_output[0])    #  只取4个月以内的数据
data_es_output = mysql.ExecQuery(sql_1)

sql = "SELECT opeTime FROM know_es_mobile_output WHERE yn=1 and input is not null and (input REGEXP '[^0-9.]')=1"  #  and opeTime LIKE '2020%'
data_es_opeTime = mysql.ExecQuery(sql)

sql_2 = """SELECT input, primaryQues
        FROM know_es_mobile_click
        WHERE yn = 1
            AND input IS NOT NULL
            AND input REGEXP '[^0-9.]' = 1
            AND TIMESTAMPDIFF(MONTH, opeTime,now()) <= 6"""
data_es_click = mysql.ExecQuery(sql_2)

print("执行SQL取主问题知识...")

sql_3 = """SELECT knowledge_id, category_id, create_time, update_time, primary_question
        FROM oc_knowledge_management
        WHERE yn = 1 
            AND `status` IN (1,2,4)
            AND base_code is not null
            AND base_code <> "XIANLIAOBASE"
            and base_code <> 'XHTXBASE'
            and base_code <> 'LXHITDH'
        """
data_es_cat = mysql.ExecQuery(sql_3)

# print("执行SQL取B端坐席的搜索日志...")

# sql_4 = """SELECT searchContent,{}
#         FROM know_seat_result
#         WHERE searchContent IS NOT NULL
#             AND searchContent REGEXP '[^0-9.]' = 1""".format(col_name_output[1])
# a = mysql.ExecQuery(sql_4)

# b = mysql.ExecQuery(
#     "select searchTime from know_seat_result where searchContent is not null and (searchContent REGEXP '[^0-9.]')=1"
# )

# sql_5 = """SELECT searchContent, primaryQuestion
#         FROM know_seat_answer
#         WHERE searchContent IS NOT NULL
#             AND searchContent REGEXP '[^0-9.]' = 1"""
# c = mysql.ExecQuery(sql_5)

# data_es_output = np.append(data_es_output, a, axis=0)
# data_es_opeTime = np.append(data_es_opeTime, b, axis=0)
# data_es_click = np.append(data_es_click, c, axis=0)


# 数据筛选函数
def judge(inp):
    inp = inp.strip()
    # 删掉以字母结尾的数据
    flag1 = 0 if re.findall(r'[a-z]$', inp) else 1
    # 无意义单字Query过滤
    flag2 = 1 if inp and len(inp)>1 else 0
    # 只保留有点击行为的数据
    flag3 = 1 if inp in click_input_list else 0
    
    if flag1 and flag2 and flag3:
        return 1
    else:
        return 0


print("清洗数据...费时...")
click_input_list = list(set([x[0] for x in data_es_click]))

data_es_output = [item for item in tqdm(data_es_output) if judge(item[0])]
data_es_output = np.array(data_es_output)
data_es_output[:, [0, 1]] = list(
    map(clean_query_and_doc, tqdm(data_es_output[:, [0, 1]]))
)
data_es_click = np.array(data_es_click)
data_es_click[:, [0, 1]] = list(
    map(clean_query_and_doc, tqdm(data_es_click[:, [0, 1]]))
)
input2doc = store_data_in_dictionary(data_es_output)

print("获取统计特征...")
ctr, clicks, views, ctr_pair = get_click_ratio(data_es_output, data_es_click)
time_feats, cat_feats = get_time_category(data_es_cat)  # 时间差 类别

print("组装用于生成向量的生语料")
sen_list = list(set(input2doc.keys())) + list(
    set([item[0] for x in list(input2doc.values()) for item in x])
)  # 未分词
sen_list = list(filter(None, sen_list))
raw_documents = [" ".join(jieba.lcut(x)) for x in sen_list]  # 已分词


sen2tfidf = {}
if "score_tfidf" in header:
    print("tfidf训练")
    tfidf_vectorizer = ut.VSM.tfidf(raw_documents, stpwrdlst)
    tfidf_vec_array = tfidf_vectorizer.transform(raw_documents).toarray()
    assert len(tfidf_vec_array) == len(sen_list)
    for i in range(len(sen_list)):
        sen = sen_list[i]
        sen2tfidf[sen] = tfidf_vec_array[i]
    tfidf_vectorizer, tfidf_vec_array = None, None

sen2bow = {}
if "score_bow" in header:
    print("bow训练")
    bow_vectorizer = ut.VSM.bow(raw_documents, stpwrdlst)
    bow_vec_array = bow_vectorizer.transform(raw_documents).toarray()
    assert len(bow_vec_array) == len(sen_list)
    for i in range(len(sen_list)):
        sen = sen_list[i]
        sen2bow[sen] = bow_vec_array[i]
    bow_vec_array = []

bm25 = None
if "score_bm25" in header:
    print("bm25训练")
    bm25 = ut.VSM.BM25(stpwrdlst=stpwrdlst)
    bm25.fit(raw_documents)

sen2w2v = {}
if "score_w2v" in header:
    print("w2v-avg句向量")
    embedding = ut.EmbeddingHelper.readTxtEmbedFile("", 200)
    for s1 in raw_documents:
        v1 = []
        s1 = s1.split(" ")
        for w in s1:
            if w not in stpwrdlst and w in embedding:
                v1.append(embedding[w])
            else:
                # 忽略OOV的情况
                v1.append(embedding["UNKNOWN_TOKEN"])
        v1 = np.array(v1).mean(axis=0)
        sen2w2v["".join(s1)] = v1
    embedding = None

sen2docvec = {}
if "score_doc2vec" in header:
    print("doc2vec训练......")
    model = ut.VSM.doc2vec(raw_documents, stpwrdlst)
    sen2docvec = {sen_list[i]: model.docvecs[i] for i in range(len(raw_documents))}

sen2bertvec = {}
if "score_bert" in header:
    print("sbert句向量获取....")
    sen2bertvec, _ = helpers.get_sbertvec(sen_list, SBERT_CONFIG["url"])
    assert set(sen_list).issubset(list(sen2bertvec.keys()))

sen_list = None
raw_documents = None

helpers.mem_check()
print("转换成pandas数据格式，并写入Excel...")
res = []  # 用于组装一行数据
for query, features in tqdm(input2doc.items()):
    score_obj_dict = []
    # 提前计算query-doc的余弦相似度
    if sen2tfidf:
        score_tfidf = cosine_similarity(
            [sen2tfidf[query]], [sen2tfidf[p[0]] for p in features]
        )
        score_tfidf = score_tfidf.tolist()[0]
        score_obj_dict.append(score_tfidf)
    if sen2bow:
        score_bow = cosine_similarity(
            [sen2bow[query]], [sen2bow[p[0]] for p in features]
        )
        score_bow = score_bow.tolist()[0]
        score_obj_dict.append(score_bow)
    if sen2docvec:
        score_doc2vec = cosine_similarity(
            [sen2docvec[query]], [sen2docvec[p[0]] for p in features]
        )
        score_doc2vec = score_doc2vec.tolist()[0]
        score_obj_dict.append(score_doc2vec)
    if sen2bertvec:
        score_bert = cosine_similarity(
            [sen2bertvec[query]], [sen2bertvec[p[0]] for p in features]
        )
        score_bert = score_bert.tolist()[0]
        score_obj_dict.append(score_bert)
    if bm25:
        docs = [" ".join(jieba.lcut(x[0])) for x in features]
        score_bm25 = bm25.transform(" ".join(jieba.lcut(query)), docs).tolist()
        score_obj_dict.append(score_bm25)

    for i in range(len(features)):
        feat = list(map(helpers.fillna, features[i]))  # col_name_output[0] + time
        primary = feat[0]
        kid = feat[1]
        res.append([])  # 用于组装一行数据

        # 根据quey-doc点击率来近似labels
        label = get_label(query, primary)
        
        res[-1].append(label)
        res[-1].append(query)

        # 点击相关特征（doc点击率、点击数、曝光数）
        feat_statistics_dict = [ctr, clicks, views]
        for obj in feat_statistics_dict:
            if obj:
                res[-1].append(obj.get(primary, 0))
        # 字符串形式的cosine相似度
        res[-1].append(round(float(cosine.similarity(query, primary)), 5))
        # 相似度得分特征
        for obj in score_obj_dict:
            if obj:
                res[-1].append(round(float(obj[i]), 5))
        # if ctr:
        #     res[-1].append(ctr.get(primary, 0))
        # if clicks:
        #     res[-1].append(clicks.get(primary, 0))
        # if views:
        #     res[-1].append(views.get(primary, 0))
        # if 1:  # 字符串形式的cosine相似度
        #     score1 = round(float(cosine.similarity(query, primary)), 5)
        #     res[-1].append(score1)
        # if score_tfidf:
        #     score2 = round(float(score_tfidf[i]), 5)
        #     res[-1].append(score2)
        # if score_bow:
        #     score3 = round(float(score_bow[i]), 5)
        #     res[-1].append(score3)
        # if score_bm25:
        #     score4 = round(float(score_bm25[i]), 5)
        #     res[-1].append(score4)
        # if score_doc2vec:
        #     score5 = round(float(score_doc2vec[i]), 5)
        #     res[-1].append(score5)
        # if score_bert:
        #     score6 = round(float(score_bert[i]), 5)
        #     res[-1].append(score6)

        # primaryQues, answerId, baseCode, botCode, searchOrder, opTime 等特征
        for x in feat:
            res[-1].append(str(x))
        # 时间差特征
        if time_feats:
            res[-1].append(time_feats.get(kid, 0))
        # 类别id特征
        if cat_feats:
            res[-1].append(cat_feats.get(kid, "unknown"))


print("写入Excel...")
print("res.shape: ({} {})\n".format(len(res), len(res[0])))  # (41432, 12)
ut.DataPersistence.writeExcel(setting.data_bucketB + setting.ltr_data_excel, res, header)
helpers.mem_check()


# 保存必要的中间变量(用于预测)
if "ctr" in header:
    joblib.dump((ctr, clicks, views), setting.data_bucketB + setting.ctr_file)
if "score_tfidf" in header:
    joblib.dump(tfidf_vectorizer, setting.data_bucketB + setting.tfidf_vec_file)
if "score_bow" in header:
    joblib.dump(bow_vectorizer, setting.data_bucketB + setting.bow_vec_file)
if "score_bm25" in header:
    joblib.dump(bm25, setting.data_bucketB + setting.bm25_file)
    # joblib.dump(model, setting.data_bucketB + setting.doc2vec_file)
if "score_doc2vec" in header:
    model.save(setting.data_bucketB + setting.doc2vec_file)  # save dov2vec
    # model.save_word2vec_format(setting.data_bucketB + setting.word2vec_file, binary=False) #save word2vec
    joblib.dump(sen2docvec, setting.data_bucketB + setting.sen2docvec_dict_file)

joblib.dump((time_feats, cat_feats), setting.data_bucketB + setting.time_feats_file)
