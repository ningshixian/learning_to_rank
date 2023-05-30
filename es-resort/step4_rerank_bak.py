# -*- coding: utf-8 -*-

import sys
import os
import codecs
import time
import re
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
import torch
torch.set_num_threads(1)  # 线程并行计算时候所占用的线程数，这个可以用来限制 PyTorch 所占用的 CPU 数目；

import helpers
import setting

sys.path.append(r"../../")
import utils_toolkit as ut
import yg_jieba as jieba

"""
uvicorn predict:app --host 0.0.0.0 --port 8099 --workers 1 --limit-concurrency 10 --reload
"""


app = FastAPI()
max_page_size = 50
cate_cols = []
DB_CONFIG, SBERT_CONFIG = helpers.get_config_info()  # 读取配置文件
stpwrdlst = ut.StringHelper.get_stopword("data/中文停用词表.txt")  # 停用词
logger = ut.LogDecorator.init_logger(setting.log_file)  # 日志封装类

# # 查询数据库获取【主问题+知识ID+类别ID+答案】
# mysql = ut.MySQLHelper.PooledDBConnection(DB_CONFIG)  # 数据库连接对象
# primary2kid, kid2cid, kid2answer = {}, {}, {}
# sql = f"""
#     SELECT 
#         a.primary_question, a.knowledge_id, a.category_id, b.answer
#     FROM
#         oc_knowledge_management a
#         LEFT JOIN oc_answer_new b ON a.knowledge_id=b.knowledge_id
#     WHERE
#         a.yn = 1 and b.yn=1
# """
# data = mysql.ExecQuery(sql)
# for item in data:
#     # 所有int类型的id都要转成字符串
#     pri, kid, cid, ans = str(item[0]), str(item[1]), str(item[2]), str(item[3])
#     primary2kid[pri] = kid
#     kid2cid[kid] = cid
#     kid2answer[kid] = ans

# 查询数据库获取【主问题+知识ID】
mysql = ut.MySQLHelper.PooledDBConnection(DB_CONFIG)  # 数据库连接对象
primary2kid = {}
sql = f"""SELECT primary_question, knowledge_id FROM oc_knowledge_management WHERE yn = 1
"""
data = mysql.ExecQuery(sql)
for item in data:
    # 所有int类型的id都要转成字符串
    pri, kid = str(item[0]), str(item[1])
    primary2kid[pri] = kid

print("加载预生成的sbert向量....")
sen2bertvec = {}
if os.path.exists(setting.bert_txt_file):
    with codecs.open(setting.bert_txt_file, "r", "utf-8") as f:
        for line in f:
            line = line.split("\t")
            sen = line[0]
            vec = list(map(lambda x: float(x), line[1].split()))
            sen2bertvec[sen] = vec

# load model with pickle to predict
with codecs.open(setting.best_trial, "r", "utf-8") as f:
    best_trial_number = str(f.read())
with codecs.open(setting.best_iteration.format(best_trial_number), "r", "utf-8") as f:
    best_iteration = int(f.read())
with open(setting.gbdt_model.format(best_trial_number), "rb") as fin:
    gbm = pkl.load(fin)

with codecs.open(setting.feature_all_file, "r", "utf-8") as f:
    feature_name_all = f.read().split("\n")
    print("所有可用特征: ", feature_name_all)  # ['time_diff', 'score', 'score_bert']
cate_cols = joblib.load(setting.cat_feat_file)  # 倒数几个特征是属于类别特征
cate_cols = list(range(cate_cols, 0))
cate_cols = [feature_name_all[idx] for idx in cate_cols][::-1]
feat_name_categ = cate_cols    # 跟 preprocessing.py 保持同步!!!
print("其中{}是类别特征".format(cate_cols))

# 加载特征对象
stdsc = joblib.load(setting.stdsc_file)  # 特征缩放模型
feat_name_scale = ["ctr", "clicks", "views", "score_bm25", "time_diff"]  # 需特征缩放的(固定)
feats_need_scale = list(set(feat_name_scale).intersection(set(feature_name_all)))  # 求交集
feats_idx_scale = [feature_name_all.index(x) for x in feats_need_scale]  # 需缩放的特征下标
print("需缩放的特征下标", feats_idx_scale)  # [0]
label_encoders = joblib.load(setting.label_encoder_file)  # 分类特征转换字典
feat_map = {
    feat_name_categ[i]:label_encoders[i]
    for i in range(len(feat_name_categ))
}
if "score_tfidf" in feature_name_all:
    tfidf_vectorizer = joblib.load(setting.tfidf_vec_file)  # tfidf字典
if "score_bow" in feature_name_all:
    bow_vectorizer = joblib.load(setting.bow_vec_file)  # bow字典
if "score_bm25" in feature_name_all:
    bm25_vectorizer = joblib.load(setting.bm25_file)  # bm25字典
if "score_doc2vec" in feature_name_all:
    # doc2vec_vectorizer = joblib.load(setting.doc2vec_file) 
    doc2vec_vectorizer = Doc2Vec.load(setting.doc2vec_file)
    sen2docvec = joblib.load(setting.sen2docvec_dict_file)  # doc2vec字典
if "time_diff" in feature_name_all:
    time_feats, cat_feats = joblib.load(setting.time_feats_file)  # {kid:time}

# 需提前分词的特征列表
feature_cut_list = [
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
    result: Union[list, dict] = {}  # Union[X, Y] 代表要么是 X 类型，要么是 Y 类型
    msg: str = "failed"  # "succeed" / "failed"
    knowledgeList: list = []  # Optional[X] 等价于 Union[X, None]
    maxScore: Union[int, float, None] = 0  # 对话联想输入

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
                + f"点调Request: {kwargs['request'].url}"
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
        flag = "龙小湖对话" if isinstance(kwargs["item2"].result, list) else "es帮助中心"
        logger.info("=== /resort/searchByScreeningImagine : {} ===".format(flag))
        logger.info(f"image_Request: {kwargs['request'].url}")
        logger.info(f"image_Query: {kwargs['item2'].query}")

        item_result = kwargs["item2"].result
        logger.info("image原始输入：\n" + "\n".join([item for item in item_result]))

        try:
            rsp = func(*args, **kwargs)
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
                + f"image_Request: {kwargs['request'].url}"
                + "\n"
                + f"image错误信息: {repr(e)}",
                setting.url,
            )
            # 服务报错时，返回原始ES结果
            rsp = kwargs["item2"].dict()
            rsp.pop("query")
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


def doc2vec_infer(text_cut):
    tmp = re.sub(" ", "", text_cut)
    if tmp in sen2docvec:
        vec = sen2docvec[tmp]
    else:
        """
        gensim的Doc2Vec模型在对新输入的句子进行推理时，会根据给定的学习率alpha重新对已训练好的权值进行迭代微调，
            以使向量更好地作为用于预测文本单词的doc-vector，这会导致预测句向量会变化！
            如果模型已经过充分训练，并且如上所述调整推理以获得更好的结果，则相同文本的向量应非​​常非常接近。
            目前来看，应该是模型训练还不够充分，加大epoch重训。
        暂时的解决方案：
        1、alpha设为0.0，则不会发生任何训练/推断（未生效！）
        2、暂存新生成的doc向量至sen2docvec！√
        可能出现的问题：
        sen2docvec字典占用的内存会持续增多！
        """
        word_list = text_cut.split(" ")
        word_list = ut.StringHelper.stopword_filter(word_list, stpwrdlst)
        vec = doc2vec_vectorizer.infer_vector(word_list, alpha=0.025, steps=50)
        sen2docvec[tmp] = vec
    return vec


def resort_func(query, result_list, knowledge_list):
    """封装-重排序函数
    
    Parameters
    query: str
    result_list: list[dict]
    knowledge_list: list[str]
    
    Return:
    重排序后的item列表以及knowledgeList
    """
    result = [x.copy() for x in result_list]  # [{}]的深拷贝!!!防止实参被修改

    # 数据清洗
    query = helpers.clean(query, stpwrdlst)
    for item in result:
        item["primary"] = helpers.clean(item["primary"], stpwrdlst)

    # VSM相似度计算，需对句子进行分词
    if set(feature_cut_list).intersection(set(feature_name_all)):
        print("============ 执行提前分词！=============")
        query_cut = " ".join(jieba.lcut(query))
        primary_cut_list = [" ".join(jieba.lcut(x["primary"])) for x in result]
        if "score_tfidf" in feature_name_all:
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
    if "score_bert" in feature_name_all:
        vec_q = helpers.request_sbert(SBERT_CONFIG["url"], query)  # 默认超时时间5s
        vec_d = []
        for x in result:
            if x["primary"] in sen2bertvec:
                vec_d.append(sen2bertvec[x["primary"]])
            else:
                logger.info("bert向量不存在:\t" + x["primary"])
                vec_d.append(
                    helpers.request_sbert(SBERT_CONFIG["url"], x["primary"])[0]
                )
        bert_scores = cosine_similarity(vec_q, vec_d)[0]

    # 特征抽取，构造输入数据格式
    X = []  # [n_samples, n_features]
    for i in range(len(result)):
        primary = result[i]["primary"]
        searchOrder = result[i]["searchOrder"]
        knowledge_id = result[i]["knowledge_id"]
        answer = result[i]["answer"]
        project_code = result[i].get("project_code", "空")    # 需要在请求参数中传入！！
        project_code = '0' if project_code=='空' else project_code
        category_id = result[i]["category_id"]
        channel = result[i].get("channel", "LF01")   # 需要在请求参数中传入！！

        line = []
        for k, feat_name in enumerate(feature_name_all):
            if "time_diff" == feat_name:
                line.append(time_feats.get(str(knowledge_id), 600))  # 默认时间差为600天
            elif "answer" == feat_name:
                line.append(1 if query in re.sub("<.*>", "", answer) else 0)
            elif "score" == feat_name:
                line.append(round(cosine.similarity(query, primary), 5))
            elif "score_tfidf" == feat_name:
                line.append(round(float(tfidf_scores[i]), 5))
            elif "score_bow" == feat_name:
                line.append(round(float(bow_scores[i]), 5))
            elif "score_bm25" == feat_name:
                line.append(round(float(bm25_scores[i]), 5))
            elif "score_doc2vec" == feat_name:
                line.append(round(float(doc2vec_scores[i]), 5))
            elif "score_bert" == feat_name:
                line.append(round(float(bert_scores[i]), 5))
            # 类别特征
            elif "searchOrder" == feat_name:
                line.append((max_page_size - searchOrder) / max_page_size)
            elif "projectCode" == feat_name:
                line.append(feat_map[feat_name].get(project_code, 0))  # label_encoders[0]["unknown"]
            elif "category_id" == feat_name:
                line.append(feat_map[feat_name].get(category_id, 0))
            elif "channel" == feat_name:
                line.append(feat_map[feat_name].get(channel, 0))
            else:
                logger.info(feat_name, "不支持的特征！")
        X.append(line)

    X = np.array(X)
    if feats_idx_scale:  # 特征缩放
        X[:, feats_idx_scale] = stdsc.transform(
            X[:, feats_idx_scale]
        ).tolist()  # X[:, [3, 7]] 取numpy数组的某几列
        X[:, feats_idx_scale] = list(
            map(lambda kk: [round(x, 5) for x in kk], X[:, feats_idx_scale])
        )

    # 模型预测
    # 需要设置线程数num_threads/nthreads=1，限制模型训练时CPU的占用率！！！否则CPU爆满！！！
    # http://ponder.work/2020/01/25/lightgbm-hang-in-multi-thread/
    preds_test = gbm.predict(
        X, num_iteration=best_iteration, num_threads=1
    )  # 从最佳迭代中获得预测结果, categorical_feature=cate_cols, 
    
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

    # 针对query=primary的情况，直接将其概率置为10！
    for i, x in enumerate(preds_test):
        preds_test[i] = x if query != result[i]["primary"] else 10

    # 根据group切分预测概率并排序，最终得到每个query的doc重排结果
    # the predicted results are the relevant scores.
    # And you can use the relevant scores to get the correct ranking order results by yourself, and qid is needed in this step.
    rank = ut.Sort.resort(preds_test)
    ret_result = [{}] * len(result)
    knowledgeList = [""] * len(result)
    for i, x in enumerate(rank):
        result_list[i]["searchOrder"] = x + 1  # 修改实参
        ret_result[x] = result_list[i]
        knowledgeList[x] = knowledge_list[i]

    return ret_result, knowledgeList


@app.post("/resort")
@log_filter
def main(request: Request, item: Item):
    """es搜索/ToB搜索 - 重排序"""
    json_data = item.dict()
    query = json_data.get("query")
    result_list = json_data.get("result")
    knowledge_list = json_data.get("knowledgeList")

    # query参数输入为空''，原样返回
    if not query:
        json_data.pop("query")
        return json_data
    # query参数缺失
    if query == "Missing query parameter":
        raise ValueError("query参数缺失!")

    # result参数输入为空
    if not result_list or not knowledge_list:
        json_data.pop("query")
        json_data["result"] = []
        json_data["knowledgeList"] = []
        return json_data

    result_list_sorted, knowledge_list_sorted = resort_func(
        query, result_list, knowledge_list
    )

    return {
        "result": result_list_sorted,
        "msg": json_data.get("msg"),
        "total": json_data.get("total"),
        "totals": json_data.get("totals"),
        "maxScore": int(json_data.get("maxScore")),
        "knowledgeList": knowledge_list_sorted,
        "experiment": "gdbt+lambdarank",
    }


@app.post("/resort/searchByScreeningImagine")
@log_filter2
def image_resort(request: Request, item2: Item2):
    """帮助中心image/对话image/相关推荐 - 重排序"""
    json_data2 = item2.dict()
    query = json_data2.get("query")  # 默认""
    msg = json_data2.get("msg")  # 默认"failed"
    maxScore = json_data2.get("maxScore")
    knowledgeList = json_data2.get("knowledgeList")  # 默认[]
    result = json_data2.get("result")  # 默认{}
    # 新增一个变量，用于判断请求的来源（"对话机器人"/"搜索"）
    is_chatbot = True if isinstance(result, list) else False

    # query参数缺失
    if query == "Missing query parameter":
        raise ValueError("query参数缺失!")

    # result/query参数输入为空''
    if not result or not query:
        json_data2.pop("query")
        if is_chatbot:
            json_data2.pop("knowledgeList")
        else:
            json_data2.pop("maxScore")
        return json_data2

    if is_chatbot:  # 对话机器人的联想输入格式
        # kid2primary = {str(i): result[i] for i in list(range(len(result)))}
        kid2primary = { primary2kid.get(pri, str(i)):pri for i, pri in enumerate(result) }
    else:
        kid2primary = {v: k for k, v in result.items()}

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
        })
    # 调用重排序函数
    result_list_sorted, knowledge_list_sorted = resort_func(
        query, result_list, kid_list
    )

    # 对输出格式进行适配
    if is_chatbot:
        return {
            "result": [r["primary"] for r in result_list_sorted],
            "msg": msg,
            "maxScore": int(maxScore),
            "experiment": "gdbt+lambdarank",
        }
    else:
        return {
            "result": {r["primary"]: r["knowledge_id"] for r in result_list_sorted},
            "msg": msg,
            "knowledgeList": knowledge_list_sorted,
            "experiment": "gdbt+lambdarank",
        }
