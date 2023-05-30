# -*- coding: utf-8 -*-

import os
import sys
import re
import psutil
import pandas as pd
import codecs
import sklearn.externals.six
import numpy as np
# from DBUtils.PooledDB import PooledDB
# from DBUtils.PersistentDB import PersistentDB
import requests
import base64
import json
from tqdm import tqdm
# from configparser import RawConfigParser
# from configparser import ConfigParser
from hashlib import md5
from operator import add
from functools import reduce
import random
import setting
sys.path.append(r"../../")
import utils_toolkit as ut
import yg_jieba as jieba

random.Random().seed(515)


def walson_ctr(num_click, num_pv, z=1.96):
    """点击率威尔逊平滑[0,1)
    https://zhuanlan.zhihu.com/p/114828907"""
    p = num_click * 1.0 / num_pv
    if p <= 0:
        return 0.0

    n = num_pv
    A = p + z ** 2 / (2 * n)
    B = np.sqrt(p * (1 - p) / n + z ** 2 / (4 * (n ** 2)))
    C = z * B

    D = 1 + z ** 2 / n
    ctr = (A - C) / D
    return float(ctr)


def get_config_info(config_file=setting.cfg_file):
    """数据库源
    67测试库 → cfg["server67"]
    43生产库 → cfg["server43"]
    RDS远程测试库 → cfg["serverRDS-dev]
    RDS远程生产库 → cfg["serverRDS"]
    """
    CFG = ut.MySQLHelper.read_config(config_file)
    
    DB_CONFIG = CFG["server"]
    DB_CONFIG["user"] = base64.b64decode(b"%s" % DB_CONFIG["user"].encode()).decode(
        "utf-8"
    )
    DB_CONFIG["passwd"] = base64.b64decode(b"%s" % DB_CONFIG["passwd"].encode()).decode(
        "utf-8"
    )
    BERT_CONFIG = CFG["sbert"]
    return DB_CONFIG, BERT_CONFIG


def request_sbert(url, sen_list, timeout=4):
    """请求sbert服务"""
    headers = {'Content-Type': 'application/json'}
    d = json.dumps({"text": sen_list})
    res = requests.post(url, data=d, headers=headers, timeout=timeout)  # 5s超时
    sen_embeddings = res.json().get("sbert_vec")
    return sen_embeddings


def dingding(msg, url):
    """钉钉-告警推送"""
    headers = {"Content-Type": "application/json;charset=utf-8"}
    json_text = {
        "msgtype": "text",
        "text": {"content": "报警 " + msg},
        "at": {"atMobiles": [""], "isAtAll": False},
    }
    requests.post(url, json.dumps(json_text), headers=headers)


def clean(text, stop_words):
    text = re.sub("<[^>]*>", "", str(text)).replace('\n','')
    text = ut.StringHelper.clean(str(text))  # 去掉标点符号和不可见字符
    word_list = jieba.lcut(text)
    word_list = ut.StringHelper.stopword_filter(word_list, stop_words)
    word_list = ut.StringHelper.number_filter(word_list)
    text = "".join(word_list).replace("   ", " ").replace("  ", " ")
    return text.strip()
    # return "".join([x for x in text if x.isalnum()])


dr = re.compile(r'<[^>]+>',re.S)
def clean_html(text):
    text = str(text)
    dd = dr.sub('',text)
    dd = re.sub(r'(&nbsp;|\s)','',dd)   #去除空格
    return dd


def fillna(x):
    if x:
        return str(x)
    else:
        return "0"


def mem_check():
    info = psutil.virtual_memory()
    print("内存使用：", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, "MB")
    print("总内存：", info.total / 1024 / 1024, "MB")
    print("内存占比：", info.percent, "%")
    # print('cpu个数：', psutil.cpu_count())


def get_sbertvec(sen_list, url):
    sen2bertvec = {}
    # 加载预生成向量
    if os.path.exists(setting.bert_txt_file):
        with codecs.open(setting.bert_txt_file, "r", "utf-8") as f:
            for line in f:
                line = line.split("\t")
                sen = line[0]
                vec = list(map(lambda x: float(x), line[1].split()))
                sen2bertvec[sen] = vec
    lenth1 = len(sen2bertvec)
    print("预加载：", lenth1)
    # 查缺补漏
    sen_ignore = []
    for sen in sen_list:
        if sen and sen not in sen2bertvec:
            sen_ignore.append(sen)
    sen_ignore = list(set(sen_ignore))
    print("sen_list：", len(sen_list))
    print("遗漏：", len(sen_ignore))
    
    # 分批次SBERT
    batchs = 32     # 400 32
    for i in tqdm(range(0, len(sen_ignore), batchs)):
        sen_batch = sen_ignore[i:i+batchs]
        # timeout cannot be set to a value less than or equal to 0.
        sen_embeddings = request_sbert(url, sen_batch, timeout=120)
        for s, s_embedding in zip(sen_batch, sen_embeddings):
            sen2bertvec[s] = s_embedding
        # print("a-(a&b): ", set(sen_batch)-(set(sen_batch)&set(sen2bertvec.keys())))
        # assert set(sen_batch).issubset(list(sen2bertvec.keys()))
    sen_ignore, sen_embeddings = None, None
    flag = (len(sen2bertvec) == lenth1)
    return sen2bertvec, flag


def iter_lines(lines, has_targets=True, one_indexed=True, missing=0.0):
    """Transforms an iterator of lines to an iterator of LETOR rows.

    Each row is represented by a (x, y, qid, comment) tuple.

    Parameters
    ----------
    lines : iterable of lines
        Lines to parse.
    has_targets : bool, optional
        Whether the file contains targets. If True, will expect the first token
        of every line to be a real representing the sample's target (i.e.
        score). If False, will use -1 as a placeholder for all targets.
    one_indexed : bool, optional 特征id从1开始的转为从0开始
        Whether feature ids are one-indexed. If True, will subtract 1 from each
        feature id.
    missing : float, optional
        Placeholder to use if a feature value is not provided for a sample.

    Yields
    ------
    x : array of floats
        Feature vector of the sample.
    y : float
        Target value (score) of the sample, or -1 if no target was parsed.
    qid : object
        Query id of the sample. This is currently guaranteed to be a string.
    comment : str
        Comment accompanying the sample.

    """
    for line in lines:
        data, _, comment = line.rstrip().partition('#')
        toks = data.strip().split()
        # toks = line.rstrip()
        # toks = re.split('\s+', toks.strip())
        # print("toks: ", toks)
        # comment = "no comment"
        num_features = 0  # 统计特征个数
        x = np.repeat(missing, 8)
        y = -1.0
        if has_targets:
            y = float(toks[0].strip())  # 相关度label
            toks = toks[1:]
        # qid:1 => 1
        qid = _parse_qid_tok(toks[0].strip())

        # feature(id:value)
        for tok in toks[1:]:
            # fid, _, val = tok.strip().partition(':') # fid,_,val => featureID,:,featureValue
            fid, val = tok.split(":")  # featureID:featureValue
            fid = int(fid)
            val = float(val)
            if one_indexed:
                fid -= 1
            assert fid >= 0
            while len(x) <= fid:
                orig = len(x)
                # x=np.resize(x,(len(x) * 2))
                x.resize(len(x) * 2)
                x[orig:orig * 2] = missing
            x[fid] = val
            num_features = max(fid + 1, num_features)

        assert num_features > 0
        x.resize(num_features)

        yield (x, y, qid, comment)


def read_dataset(source, has_targets=True, one_indexed=True, missing=0.0):
    """Parses a LETOR dataset from `source`.

    Parameters
    ----------
    source : string or iterable of lines
        String, file, or other file-like object to parse.
    has_targets : bool, optional
        See `iter_lines`.
    one_indexed : bool, optional
        See `iter_lines`.
    missing : float, optional
        See `iter_lines`.

    Returns
    -------
    X : array of arrays of floats
      Feature matrix (see `iter_lines`).
    y : array of floats
        Target vector (see `iter_lines`).
    qids : array of objects
        Query id vector (see `iter_lines`).m
    comments : array of strs
        Comment vector (see `iter_lines`).
    """
    if isinstance(source, sklearn.externals.six.string_types):
        source = source.splitlines(True)

    max_width = 0  # 某行最多特征个数
    xs, ys, qids, comments = [], [], [], []
    iter_content = iter_lines(source, has_targets=has_targets,
                              one_indexed=one_indexed, missing=missing)
    # x:特征向量; y:float 相关度值[0-4]; qid:string query id; comment: #后面内容
    for x, y, qid, comment in iter_content:
        xs.append(x)
        ys.append(y)
        qids.append(qid)
        comments.append(comment)
        max_width = max(max_width, len(x))

    assert max_width > 0
    # X.shape = [len(xs), max_width]
    X = np.ndarray((len(xs), max_width), dtype=np.float64)
    X.fill(missing)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    ys = np.array(ys) if has_targets else None
    qids = np.array(qids)
    comments = np.array(comments)

    return (X, ys, qids, comments)


def _parse_qid_tok(tok):
    assert tok.startswith('qid:')
    return tok[4:]


def trans_to_dataframe(mat, cate_cols=[]):
    # 将libsvm转为dataframe
    if str(type(mat)) == "<class 'scipy.sparse.csr.csr_matrix'>":
        mat = mat.todense()  # (167220, 12)
    df1 = pd.DataFrame(mat)
    # 声明类别特征
    for i in cate_cols:
        column_name = df1.columns[i]
        df1[column_name] = df1[column_name].astype("category")
    # print(df1.dtypes)
    return df1


def md5_hash(key):
    xx = md5()  # 导入md5算法
    xx.update(key.encode('utf8'))  # 把值传给md5算法
    return xx.hexdigest()


def mapping(hashkey, n=10):
    return reduce(add, [ord(i) for i in hashkey]) % n


def hash_bucketing(oaAccount, nb=2):
    """哈希分桶 - A/B测试
    对每个用户唯一id做hash运算，并对hash值对2取模，便可以将用户分成0，1两组
    散列函数（hash function）对一种对任意输入，都返回一个固定长度输出的函数
    参考：http://yalei.name/2018/12/split-stream
    """
    # hash_bytes = hash(oaAccount)
    # bucket = 'A' if hash_bytes%nb==1 else 'B'
    hashkey = md5_hash(oaAccount)
    bucket = 'A' if mapping(hashkey, nb) else 'B'
    if 'ningshixian' in oaAccount:
        bucket = 'A'
    return bucket
