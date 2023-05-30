# -*- coding: utf-8 -*-

import sys
import os
import codecs
import time
import re
import math
import pickle as pkl
import traceback
from functools import wraps
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import threading
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from textdistance import cosine
from gensim.models.doc2vec import Doc2Vec
from fastapi import FastAPI
from starlette.requests import Request
from pydantic import BaseModel, validator
# import lightgbm as lgb
# import uvicorn
# from sentence_transformers import SentenceTransformer
import Levenshtein
from textdistance import cosine, jaccard, Hamming
import torch
torch.set_num_threads(1)  # 线程并行计算时候所占用的线程数，这个可以用来限制 PyTorch 所占用的 CPU 数目；

import helpers
import setting

sys.path.append(r"../../")
import utils_toolkit as ut
import yg_jieba as jieba

"""
uvicorn step4_rerank:app --host 0.0.0.0 --port 9908 --workers 1 --limit-concurrency 50 --reload
"""


app = FastAPI()
max_page_size = 50
cate_cols = []
DB_CONFIG, SBERT_CONFIG = helpers.get_config_info()  # 读取配置文件
stpwrdlst = ut.StringHelper.get_stopword(setting.stopword_file)  # 停用词
logger = ut.LogDecorator.init_logger(setting.log_file)  # 日志封装类

# 数据库连接对象
mysql = ut.MySQLHelper.PooledDBConnection(DB_CONFIG)  

# # 查询数据库获取【主问题+知识ID+关键词】
# primary2kid, kid2kw = {}, {}
# sql = """SELECT primary_question, knowledge_id, key_word
#         FROM oc_knowledge_management
#         WHERE yn = 1
#             AND status IN (1, 2, 4, 5)"""
# data = mysql.ExecQuery(sql)
# for item in data:
#     # 所有int类型的id都要转成字符串
#     pri, kid, kw = str(item[0]), str(item[1]), str(item[2])
#     primary2kid[pri] = kid
#     kid2kw[kid] = kw

# # 获取主问题对应的答案
# sql_1 = """
#     SELECT
#         cc.knowledge_id, cc.primary_question, d.answer
#     FROM
#         oc_knowledge_management cc
#         INNER JOIN oc_answer_mapping b ON cc.knowledge_id = b.knowledge_id
#         LEFT JOIN oc_answer_new d ON b.answer_id = d.answer_id 
#     WHERE
#         cc.STATUS IN ( 1, 2, 4 ) 
#         AND cc.yn = 1
#         AND cc.base_code is not null
#         AND cc.base_code <> "XIANLIAOBASE"
#         AND cc.base_code <> 'XHTXBASE'
#         AND cc.base_code <> 'LXHITDH'
#         AND d.yn = 1 
#         AND d.channel_flag LIKE '%在线%' 
#         AND b.yn = 1 
#     """
# tmp = mysql.ExecQuery(sql_1)
# pri_answer = {}
# for item in tmp:
#     key = helpers.clean(item[1], stpwrdlst)
#     value = [item[2], item[0]]
#     if not key in pri_answer:
#         pri_answer[key] = value


print("加载预生成的sbert向量....")
sen2bertvec = {}
if os.path.exists(setting.bert_txt_file):
    with codecs.open(setting.bert_txt_file, "r", "utf-8") as f:
        for line in f:
            line = line.split("\t")
            sen = line[0]
            vec = list(map(lambda x: float(x), line[1].split()))
            sen2bertvec[sen] = vec
            assert len(vec) ==768

# load model with pickle to predict
# bucketA
with codecs.open(setting.model_bucketA + setting.best_trial, "r", "utf-8") as f:
    best_trial_number = str(f.read())
with codecs.open(setting.model_bucketA + setting.best_iteration.format(best_trial_number), "r", "utf-8") as f:
    best_iteration_A = int(f.read())
with open(setting.model_bucketA + setting.gbdt_model.format(best_trial_number), "rb") as fin:
    gbm_A = pkl.load(fin)
with codecs.open(setting.data_bucketA + setting.feature_all_file, "r", "utf-8") as f:
    feature_name_all_A = f.read().split("\n")
    print("模型A绑定特征: ", feature_name_all_A)  # ['time_diff', 'score', 'score_bert']
# bucketB
with codecs.open(setting.model_bucketB + setting.best_trial, "r", "utf-8") as f:
    best_trial_number = str(f.read())
with codecs.open(setting.model_bucketB + setting.best_iteration.format(best_trial_number), "r", "utf-8") as f:
    best_iteration_B = int(f.read())
with open(setting.model_bucketB + setting.gbdt_model.format(best_trial_number), "rb") as fin:
    gbm_B = pkl.load(fin)
with codecs.open(setting.data_bucketB + setting.feature_all_file, "r", "utf-8") as f:
    feature_name_all_B = f.read().split("\n")
    print("模型B绑定特征: ", feature_name_all_B)  # ['time_diff', 'score', 'score_bert']


# load feature obj
# bucketA
if "ctr" in feature_name_all_A:
    ctr_pair, clicks_pair, views_pair = joblib.load(setting.data_bucketA + setting.ctr_file)
if "time_diff" in feature_name_all_A:
    time_feats_A = joblib.load(setting.data_bucketA + setting.time_feats_file)  # {kid:time}
if "tfidf_cos_sim" in feature_name_all_A:
    tfidf_vectorizer_A = joblib.load(setting.data_bucketA + setting.tfidf_vec_file)  # tfidf字典
# load feature obj
# bucketB
# cate_cols = joblib.load(setting.data_bucketB + setting.cat_feat_file)  # 倒数几个特征是属于类别特征
# cate_cols = list(range(cate_cols, 0))
# cate_cols = [feature_name_all_B[idx] for idx in cate_cols][::-1]
feat_name_categ = cate_cols    # 跟 preprocessing.py 保持同步!!!
print("其中{}是类别特征".format(cate_cols))
stdsc = joblib.load(setting.data_bucketB + setting.stdsc_file)  # 特征缩放模型
feat_name_scale = ["ctr", "clicks", "views", "score_bm25", "time_diff"]  # 需特征缩放的(固定)
feats_need_scale = list(set(feat_name_scale).intersection(set(feature_name_all_B)))  # 求交集
feats_idx_scale = [feature_name_all_B.index(x) for x in feats_need_scale]  # 需缩放的特征下标
print("需缩放的特征下标", feats_idx_scale)  # [0]
label_encoders = joblib.load(setting.data_bucketB + setting.label_encoder_file)  # 分类特征转换字典
feat_map = {
    feat_name_categ[i]:label_encoders[i]
    for i in range(len(feat_name_categ))
}
if "score_tfidf" in feature_name_all_B:
    tfidf_vectorizer_B = joblib.load(setting.data_bucketB + setting.tfidf_vec_file)  # tfidf字典
if "score_bow" in feature_name_all_B:
    bow_vectorizer = joblib.load(setting.data_bucketB + setting.bow_vec_file)  # bow字典
if "score_bm25" in feature_name_all_B:
    bm25_vectorizer = joblib.load(setting.data_bucketB + setting.bm25_file)  # bm25字典
if "score_doc2vec" in feature_name_all_B:
    # doc2vec_vectorizer = joblib.load(setting.doc2vec_file) 
    doc2vec_vectorizer = Doc2Vec.load(setting.data_bucketB + setting.doc2vec_file)
    sen2docvec = joblib.load(setting.data_bucketB + setting.sen2docvec_dict_file)  # doc2vec字典
if "time_diff" in feature_name_all_B:
    time_feats_B, cat_feats = joblib.load(setting.data_bucketB + setting.time_feats_file)  # {kid:time}


feature_cut_list = [
    "tfidf_cos_sim", 
    "score_tfidf",
    "score_bow",
    "score_bm25",
    "score_doc2vec",
]


class Item(BaseModel):
    query: str = "Missing query parameter"
    result: list = []
    knowledgeList: list = []
    msg: str = "failed"
    total: int = 0
    totals: Optional[int] = 0  # es
    maxScore: Union[int, float, None] = 0  # toB
    oaAccount: str = "none"
    requestSource: Optional[str] = None  # 可选参数 recommendation / help / seat / robot

    @validator("totals")
    def totals_cannot_be_none(cls, v):
        return 0 if v == None else v

    @validator("maxScore")
    def maxScore_cannot_be_none(cls, v):
        if v == None:
            return 0
        elif isinstance(v, float):
            return int(v)
        else:
            return v


class Item2(BaseModel):
    query: str = "Missing query parameter"
    result: Union[list, dict] = {}
    resultAsKey: Union[list, dict] = {} # Union[X, Y] 代表要么是 X 类型，要么是 Y 类型
    msg: str = "failed"  # "succeed" / "failed"
    knowledgeList: list = []  # Optional[X] 等价于 Union[X, None]
    maxScore: Union[int, float, None] = 0  # 对话联想输入
    oaAccount: str = "none"

    @validator("maxScore")
    def maxScore_cannot_be_none(cls, v):
        if v == None:
            return 0
        elif isinstance(v, float):
            return int(v)
        else:
            return v


def log_filter(func):
    """接口日志"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = 1000 * time.time()
        logger.info(f"=============  Begin: {func.__name__}  =============")
        logger.info(f"Request: {kwargs['request'].url}")
        logger.info(f"query: {kwargs['item'].query}")
        logger.info(f"knowledgeList: {kwargs['item'].knowledgeList}")
        logger.info(f"oaAccount: {kwargs['item'].oaAccount}")
        if kwargs["item"].result:
            logger.info(
                "原始输入：\n"
                + "\n".join(
                    [
                        str(yy["searchOrder"]) + "\t" + yy["primary"]
                        for yy in kwargs["item"].result
                    ]
                )
            )

        try:
            # 执行排序操作
            rsp = func(*args, **kwargs)
            if rsp["result"]:
                logger.info(
                    "重排序结果：\n"
                    + "\n".join(
                        [
                            "Response: " + str(zz["searchOrder"]) + "\t" + zz["primary"]
                            for zz in rsp["result"]
                        ]
                    )
                )
        except Exception as e:
            """
            前台传的query是空，传到重排服务时把空的字段过滤了导致的
            """
            logger.error("异常报警：")
            logger.error(kwargs["item"])
            logger.error(traceback.format_exc())
            # 钉钉推送错误消息
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            helpers.dingding(
                str(t)
                + "\n"
                + f"es_resort_Request: {kwargs['request'].url}"
                + "\n"
                + f"Args: {kwargs['item'].query}"
                + "\n"
                + f"点调总耗时: {1000 * time.time() - start}ms"
                + "\n"
                + f"错误信息: {repr(e)}",
                setting.url,
            )
            # 服务报错时，返回原始ES结果
            rsp = kwargs["item"].dict()
            rsp.pop("query")
            rsp.pop("oaAccount")
            rsp["msg"] = "failed"

        end = 1000 * time.time()
        logger.info(f"Time consuming: {end - start}ms")
        logger.info(f"=============   End: {func.__name__}   =============\n")
        return rsp

    return wrapper


def log_filter2(func):
    """接口日志"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = 1000 * time.time()
        flag = "龙小湖对话" if isinstance(kwargs["item2"].resultAsKey, list) else "es帮助中心"
        logger.info("=== /resort/searchByScreeningImagine : {} ===".format(flag))
        logger.info(f"image_Request: {kwargs['request'].url}")
        logger.info(f"image_Query: {kwargs['item2'].query}")
        logger.info(f"knowledgeList: {kwargs['item2'].knowledgeList}")
        logger.info(f"oaAccount: {kwargs['item2'].oaAccount}")
        # logger.error(kwargs["item2"])

        item_result = kwargs["item2"].resultAsKey
        if item_result:
            tmp = item_result.values() if isinstance(item_result, dict) else item_result
            logger.info("image原始输入：\n" + "\n".join([item for item in tmp]))

        try:
            rsp = func(*args, **kwargs)
            if rsp["result"]:
                logger.info("image重排序结果：\n" + "\n".join([xx for xx in rsp["result"]]))
        except Exception as e:
            logger.error("异常报警：")
            logger.error(kwargs["item2"])
            logger.error(traceback.format_exc())
            # 钉钉推送错误消息
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            helpers.dingding(
                str(t)
                + "\n"
                + f"image_resort_Request: {kwargs['request'].url}"
                + "\n"
                + f"Args: {kwargs['item2'].query}"
                + "\n"
                + f"耗时: {1000 * time.time() - start}ms"
                + "\n"
                + f"image错误信息: {repr(e)}",
                setting.url,
            )
            # 服务报错时，返回原始ES结果
            rsp = kwargs["item2"].dict()
            rsp.pop("query")
            rsp.pop("oaAccount")
            if flag == "龙小湖对话":
                rsp.pop("knowledgeList")
            else:
                rsp.pop("maxScore")
            rsp["msg"] = "failed"

        end = 1000 * time.time()
        logger.info(f"image Time consuming: {end - start}ms")
        logger.info(f"=============   End: {func.__name__}   =============\n")
        return rsp

    return wrapper


# 统计quey-doc的词共现
def get_num_common_words_and_ratio(query, primary):
    q1_word_set = set(query)
    q2_word_set = set(primary)
    q1_word_len_set = len(q1_word_set)
    q2_word_len_set = len(q2_word_set)

    result = len(q1_word_set & q2_word_set)
    result_ratio_q_set = result / q1_word_len_set if q1_word_len_set else 0
    result_ratio_t_set = result / q2_word_len_set if q2_word_len_set else 0
    return result, result_ratio_q_set, result_ratio_t_set

# 获取某字段上与查询词匹配的分词词组个数与该字段总词组个数的比值
def get_field_match_ratio(query, primary):
    q = set(jieba.lcut(query))
    p = set(jieba.lcut(primary))
    result_ratio_t_set = len(q & p) / len(p) if p else 0
    return result_ratio_t_set

# Jaccard 相似度
def jaccard(x, y):
    if str(y)=='':
        y = '0'
    x = set(x)
    y = set(y)
    return float(len(x & y) / len(x | y))


def resort_func(bucket, query, query_ori, result_list, knowledge_list):
    """封装-重排序函数
    
    Parameters
    query: str
    result_list: list[dict]
    knowledge_list: list[str]
    
    Return:
    重排序后的item列表以及knowledgeList
    """
    result = [x.copy() for x in result_list]  # [{}]的深拷贝!!!防止实参被修改
    for item in result:  # 数据清洗
        item["primary"] = helpers.clean(item["primary"], stpwrdlst)
        item["answer"] = helpers.clean_html(item["answer"])
    
    # print(bucket)
    if bucket=='A':
        gbm = gbm_A
        best_iteration = best_iteration_A
        feature_name_all = feature_name_all_A
        if "time_diff" in feature_name_all_A:
            time_feats = time_feats_A
        if "tfidf_cos_sim" in feature_name_all_A:
            tfidf_vectorizer = tfidf_vectorizer_A
    else:
        gbm = gbm_B
        best_iteration = best_iteration_B
        feature_name_all = feature_name_all_B
        if "time_diff" in feature_name_all_B:
            time_feats = time_feats_B
        if "score_tfidf" in feature_name_all_B:
            tfidf_vectorizer = tfidf_vectorizer_B

    # VSM相似度计算，需对句子进行分词
    if set(feature_cut_list).intersection(set(feature_name_all)):
        query_cut = " ".join(jieba.lcut(query))
        primary_cut_list = [" ".join(jieba.lcut(x["primary"])) for x in result]
        if "score_tfidf" in feature_name_all or "tfidf_cos_sim" in feature_name_all:
            tfidf_vec_array = tfidf_vectorizer.transform(
                [query_cut] + primary_cut_list
            ).toarray()
            tfidf_scores = cosine_similarity([tfidf_vec_array[0]], tfidf_vec_array[1:])[
                0
            ]
        if "score_bow" in feature_name_all:
            bow_vec_array = bow_vectorizer.transform(
                [query_cut] + primary_cut_list
            ).toarray()
            bow_scores = cosine_similarity([bow_vec_array[0]], bow_vec_array[1:])[0]
        if "score_bm25" in feature_name_all:
            bm25_scores = bm25_vectorizer.transform(query_cut, primary_cut_list)
        if "score_doc2vec" in feature_name_all:
            doc2vec_scores = cosine_similarity(
                [doc2vec_infer(query_cut)], [doc2vec_infer(p) for p in primary_cut_list]
            )[0]
    
    # sbert相似度计算
    if set(["sbert_cos_sim", "score_bert"]).intersection(set(feature_name_all)):
        vec_q = helpers.request_sbert(SBERT_CONFIG["url"], query)  # 默认超时时间5s; return list
        vec_d, vec_a = [], []
        for x in result:
            # 主问题向量映射
            if x["primary"] in sen2bertvec:
                vec_d.append(sen2bertvec[x["primary"]])
                # ValueError: setting an array element with a sequence.
                # 参与轻评价调研送积分活动说明流程 → sbert向量的维度是180？ → 无法形成2D阵列
            else:
                logger.info("{} bert向量不存在！已添加到字典中".format(x["primary"]))
                tmp = helpers.request_sbert(SBERT_CONFIG["url"], x["primary"])[0]
                sen2bertvec[x["primary"]] = tmp  # 加入sbert字典
                vec_d.append(tmp)
            # 答案向量映射
            if x["answer"] in sen2bertvec:
                vec_a.append(sen2bertvec[x["answer"]])
            else:
                tmp = helpers.request_sbert(SBERT_CONFIG["url"], x["answer"])[0]
                sen2bertvec[x["answer"]] = tmp  # 加入sbert字典
                vec_a.append(tmp)
        # 1、计算query和primary的余弦相似度
        bert_scores = cosine_similarity(vec_q, vec_d)[0]
        # 2、计算query和answer的余弦相似度
        ans_bert_scores = cosine_similarity(vec_q, vec_a)[0]
        # answer_list = [helpers.clean_html(x["answer"]) for x in result]
        # vec_a = helpers.request_sbert(SBERT_CONFIG["url"], answer_list, timeout=15)
        # ans_bert_scores = cosine_similarity(vec_q, vec_a)[0]

    # 特征抽取，构造输入数据格式
    X = []  # [n_samples, n_features]
    for i in range(len(result)):
        primary = result[i]["primary"]
        searchOrder = result[i]["searchOrder"]
        knowledge_id = str(result[i]["knowledge_id"])
        answer = result[i]["answer"]
        project_code = result[i].get("project_code", "空")    # 需要在请求参数中传入！！
        project_code = '0' if project_code=='空' else project_code
        category_id = result[i]["category_id"]
        channel = result[i].get("channel", "LF01")   # 需要在请求参数中传入！！
        
        line = []
        common_feats_name = ['num_common_words', 'num_common_words_ratio1', 'num_common_words_ratio2']
        common_feats_value = list(get_num_common_words_and_ratio(query, primary))
        for k, feat_name in enumerate(feature_name_all):
            if 'ctr'==feat_name :
                line.append(ctr_pair.get((query, primary), 0))
            elif 'clicks'==feat_name:
                line.append(clicks_pair.get((query, primary), 0))
            elif 'views'==feat_name:
                line.append(views_pair.get((query, primary), 0))
            elif feat_name in common_feats_name:
                line.append(common_feats_value[common_feats_name.index(feat_name)])
            elif 'field_match_ratio' == feat_name:
                line.append(get_field_match_ratio(query, primary))
            elif 'num_jaccard_sim' == feat_name:
                line.append(jaccard(query, primary))
            elif 'num_edict_distance_k_pt' == feat_name:
                line.append(Levenshtein.distance(query, primary) / (len(x)+1))
            elif 'num_edict_jaro' == feat_name:
                line.append(Levenshtein.jaro(query, primary))
            elif 'num_edict_ratio' == feat_name:
                line.append(Levenshtein.ratio(query, primary))
            elif 'num_edict_jaro_winkler' == feat_name:
                line.append(Levenshtein.jaro_winkler(query, primary))
            elif feat_name in ['tfidf_cos_sim', "score_tfidf"]:
                line.append(round(float(tfidf_scores[i]), 5))
            elif "score_bow" == feat_name:
                line.append(round(float(bow_scores[i]), 5))
            elif "score_bm25" == feat_name:
                line.append(round(float(bm25_scores[i]), 5))
            elif "score_doc2vec" == feat_name:
                line.append(round(float(doc2vec_scores[i]), 5))
            elif feat_name in ["sbert_cos_sim", "score_bert"]:
                line.append(round(float(bert_scores[i]), 5))
            elif feat_name in ["answer_sbert_cos_sim"]:
                line.append(round(float(ans_bert_scores[i]), 5))
            elif feat_name in ["string_cosine", "score"]:
                line.append(round(cosine.similarity(query, primary), 5))
            elif "Hamming_kt" == feat_name:
                line.append(Hamming(qval=None).normalized_distance(query, primary))
            elif "Hamming_sim_kt" == feat_name:
                line.append(Hamming(qval=None).similarity(query, primary))
            elif "time_diff" == feat_name:
                line.append(time_feats.get(str(knowledge_id), 600))  # 默认时间差为600天
            elif "searchOrder" == feat_name:
                line.append(searchOrder)
            # 类别特征
            elif "projectCode" == feat_name:
                line.append(feat_map[feat_name].get(project_code, 0))  # label_encoders[0]["unknown"]
            elif "category_id" == feat_name:
                line.append(feat_map[feat_name].get(category_id, 0))
            elif "channel" == feat_name:
                line.append(feat_map[feat_name].get(channel, 0))
            else:
                logger.info(feat_name, "不支持的特征！")
        line = list(map(lambda x:round(x, 5), line))
        X.append(line)

    X = np.array(X)
    if bucket=='B' and feats_idx_scale:  # 特征缩放
        X[:, feats_idx_scale] = stdsc.transform(
            X[:, feats_idx_scale]
        ).tolist()  # X[:, [3, 7]] 取numpy数组的某几列
        X[:, feats_idx_scale] = list(
            map(lambda kk: [round(x, 5) for x in kk], X[:, feats_idx_scale])
        )

    # 模型预测
    # 需要设置线程数num_threads/nthreads=1，限制模型训练时CPU的占用率！！！否则CPU爆满！！！
    # http://ponder.work/2020/01/25/lightgbm-hang-in-multi-thread/
    # gbm.predict()输出每个样本的相关度（不一定是0到1的值 -----LambdaMart原理）
    preds_test = gbm.predict(
        X, num_iteration=best_iteration, num_threads=1
    )  # 从最佳迭代中获得预测结果, categorical_feature=cate_cols,
    # print(X)
    # print(preds_test)
    
    # # GBDT和LR的融合模型预测
    # y_pred = gbm.predict(X, num_iteration=best_iteration, num_threads=1, pred_leaf=True)
    # with open('model/LR.mod', "rb") as fin:
    #     lm = pkl.load(fin)
    # num_leaf = 465
    # transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
    # for i in range(0, len(y_pred)):
    #     temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    #     transformed_testing_matrix[i][temp] += 1
    # preds_test = lm.predict(transformed_testing_matrix)   # Give the probabilty on each label

    # # 针对query=primary或query in sims的情况，直接将其概率置为max_score！
    # max_score = math.ceil(max(preds_test))
    # for i, x in enumerate(preds_test):
    #     if query_ori in result[i]["similar_ques"] or query == result[i]["primary"]:
    #         # 主问题已被清洗，而相似问未经过处理
    #         preds_test[i] = max_score
    #     else:
    #         preds_test[i] = x

    # # 根据group切分预测概率并排序，最终得到每个query的doc重排结果
    # # the predicted results are the relevant scores.
    # # And you can use the relevant scores to get the correct ranking order results by yourself, and qid is needed in this step.
    # rank = ut.Sort.resort(preds_test)
    # ret_result = [{}] * len(result)
    # knowledgeList = [""] * len(result)
    # for i, x in enumerate(rank):
    #     result_list[i]["searchOrder"] = x + 1  # 修改实参
    #     ret_result[x] = result_list[i]
    #     knowledgeList[x] = knowledge_list[i]

    # return ret_result, knowledgeList
    
    # 按预测概率重排原始结果
    rank = ut.Sort.resort(preds_test)   # relevant scores:[0.8,0.2,0.6,0.3] → ranking idx:[0,2,3,1] → ranking order:[0,3,1,2]
    r = [{}] * len(result)
    k = [""] * len(result)
    for i, x in enumerate(rank):    # 第 i 位排第 x 名
        r[x] = result_list[i]
        k[x] = knowledge_list[i]
    
    # 后处理
    new_ind = sorted(range(len(r)),     # 多条件同时排序
        key=lambda x: (
            query not in r[x]["primary"], 
            query_ori not in r[x]["similar_ques"],
            query not in r[x]["answer"] 
        )
    )
    r = [r[i] for i in new_ind]
    k = [k[i] for i in new_ind]
    for i,res in enumerate(r):
        res["searchOrder"] = i + 1  # 修改实参
    
    return r, k



@app.post("/resort")
@log_filter
def main(request: Request, item: Item):
    """es搜索/ToB搜索 - 重排序"""
    json_data = item.dict()
    query_ori = json_data.get("query")
    result_list = json_data.get("result")
    knowledge_list = json_data.get("knowledgeList")
    oaAccount = json_data.get("oaAccount")
    requestSource = json_data.get("requestSource")
    
    # 数据清洗
    query = helpers.clean(query_ori, stpwrdlst)

    # # query参数输入为空''，原样返回
    # if not query:
    #     json_data.pop("query")
    #     json_data.pop("oaAccount")
    #     return json_data
    
    # # query参数缺失
    # if query == "Missing query parameter":
    #     raise ValueError("query参数缺失!")

    # query参数输入为空'' 或 query参数缺失，原样返回
    if not query or query == "Missing query parameter":
        json_data.pop("query")
        json_data.pop("oaAccount")
        json_data.pop("requestSource")
        json_data["msg"] = "failed"
        return json_data

    # result参数输入为空
    if not result_list or not knowledge_list:
        json_data.pop("query")
        json_data.pop("oaAccount")
        json_data.pop("requestSource")
        json_data["result"] = []
        json_data["knowledgeList"] = []
        return json_data
    
    # 坐席源不进行排序，直接返回
    if requestSource and requestSource == "seat":
        json_data.pop("query")
        json_data.pop("oaAccount")
        json_data.pop("requestSource")
        return json_data

    if oaAccount in ['none', 'test']:
        bucket = 'A'
    else:
        bucket = helpers.hash_bucketing(oaAccount)  # 哈希分流
    logger.info("哈希分流: " + bucket)
    result_list_sorted, knowledge_list_sorted = resort_func(
        bucket, query, query_ori, result_list, knowledge_list
    )

    return {
        "result": result_list_sorted,
        "msg": json_data.get("msg"),
        "total": json_data.get("total"),
        "totals": json_data.get("totals"),
        "maxScore": int(json_data.get("maxScore")),
        "knowledgeList": knowledge_list_sorted,
        "experiment": bucket,   # "gdbt+lambdarank"
    }


@app.post("/resort/searchByScreeningImagine")
@log_filter2
def image_resort(request: Request, item2: Item2):
    """帮助中心image/对话image/相关推荐 - 重排序"""
    json_data2 = item2.dict()
    query_ori = json_data2.get("query")  # 默认""
    msg = json_data2.get("msg")  # 默认"failed"
    maxScore = json_data2.get("maxScore")
    knowledgeList = json_data2.get("knowledgeList")  # 默认[]
    result = json_data2.get("result")  # 默认{}
    resultAsKey = json_data2.get("resultAsKey")  # 默认{}
    if not resultAsKey:
        resultAsKey = result
    oaAccount = json_data2.get("oaAccount")
    
    # 新增一个变量，用于判断请求的来源（"对话机器人"/"搜索"）
    is_chatbot = True if isinstance(resultAsKey, list) else False

    # 数据清洗
    query = helpers.clean(query_ori, stpwrdlst)

    # # query参数缺失
    # if query == "Missing query parameter":
    #     raise ValueError("query参数缺失!")

    # query参数缺失
    if query == "Missing query parameter":
        json_data2.pop("query")
        json_data2.pop("oaAccount")
        if is_chatbot:
            json_data2.pop("knowledgeList")
        else:
            json_data2.pop("maxScore")
        json_data2["msg"] = "failed"
        return json_data2
        # raise ValueError("query参数缺失!")

    # resultAsKey/query参数输入为空''
    if not resultAsKey or not query:
        json_data2.pop("query")
        json_data2.pop("oaAccount")
        if is_chatbot:
            json_data2.pop("knowledgeList")
        else:
            json_data2.pop("maxScore")
        return json_data2

    kid2primary = {}
    if is_chatbot:  # 对话机器人的联想输入格式
        kid2primary = { str(i):pri for i, pri in enumerate(resultAsKey) }
    else:
        kid2primary = resultAsKey.copy()

    # 组装新格式的result数据
    result_list = []
    kid_list = list(kid2primary.keys())  # 以result中的id列表为准
    for i, kid in enumerate(kid_list):
        result_list.append({
            "knowledge_id": kid, 
            "primary": kid2primary[kid],    # kid2primary.get(kid, '')
            "searchOrder": i + 1,
            "answer": '',
            "category_id": '',
            "similar_ques": '',
        })

    # kid2primary = {}
    # if is_chatbot:  # 对话机器人的联想输入格式
    #     for i, pri in enumerate(resultAsKey):
    #         answer, kid = pri_answer.get(helpers.clean(pri, stpwrdlst), ['None', i])
    #         kid2primary[kid] = [pri, answer]
    # else:
    #     for kid, pri in resultAsKey.items():
    #         answer, kid = pri_answer.get(helpers.clean(pri, stpwrdlst), ['None', kid])
    #         kid2primary[kid] = [pri, answer]

    # # 组装新格式的result数据
    # result_list = []
    # kid_list = list(kid2primary.keys())  # 以result中的id列表为准
    # for i, kid in enumerate(kid_list):
    #     pri, answer = kid2primary[kid]
    #     result_list.append({
    #         "knowledge_id": str(kid), 
    #         "primary": pri,
    #         "searchOrder": i + 1,
    #         "answer": answer,
    #         "category_id": '',
    #         "similar_ques": '',
    #     })
    
    # 调用重排序函数
    if oaAccount in ['none', 'test']:
        bucket = 'A'
    else:
        bucket = helpers.hash_bucketing(oaAccount)  # 哈希分流
    logger.info("哈希分流: " + bucket)
    result_list_sorted, knowledge_list_sorted = resort_func(
        bucket, query, query_ori, result_list, kid_list
    )

    # 对输出格式进行适配
    if is_chatbot: # []
        return {
            "result": [r["primary"] for r in result_list_sorted],
            "msg": msg,
            "maxScore": int(maxScore),
            "experiment": bucket,   # "gdbt+lambdarank"
        }
    else:   # {}
        tmp_result = {}
        for r in result_list_sorted:    # 相同主问题，通过在后面加空格进行区分
            while r["primary"] in tmp_result:
                r["primary"] = r["primary"]+' '
            tmp_result[r["primary"]] = r["knowledge_id"]
        return {
            "result": tmp_result,
            "msg": msg,
            "knowledgeList": knowledge_list_sorted,
            "experiment": bucket,   # "gdbt+lambdarank"
        }
