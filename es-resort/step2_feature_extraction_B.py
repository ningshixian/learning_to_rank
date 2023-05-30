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
import setting
import helpers

sys.path.append(r"../../")
import utils_toolkit as ut

"""
5、对抽取的特征值进行放缩-标准化
6、构造 libsvm 训练数据集，特征如下：
        1. score
        2. score_tfidf 
        3. score_bow 
        4. score_bm25
        5. score_doc2vec
        6. score_bert
        7. searchOrder
        8. time_diff
        9. edit_distance
    PS:相同query的qid应相同 
7、划分训练/验证/测试集
"""

max_page_size = 50
feat_name_scale = ["ctr", "clicks", "views", "score_bm25", "time_diff"]  # 需特征缩放的(固定)
feat_name_categ = []  # 需转换为类别特征的(若不使用置为空[])
# feat_name_categ = ["knowledgeId", "answerId", "category_id", "channel", "projectCode"]\
feat_name_other = ["score", "score_bert"]   # 其他想要添加的特征(除了上面的), "score_doc2vec", "searchOrder"


def feature_scaling(feat_list):
    """为什么要进行特征缩放呢？
    有些特征的值是有区间界限的，如年龄，体重。而有些特征的值是可以无限制增加，如计数值。
    特征与特征之间数值的差距会对模型产生不良影响
    """
    stdsc = MinMaxScaler()  # 归一化之后所有数据被映射到0-1区间内
    # stdsc = StandardScaler()  # 标准化后，所有数据的均值为0，方差为1(以列为基准)

    stdsc = stdsc.fit(feat_list)  # 训练 [n_samples, n_features]
    tmp_data = stdsc.transform(feat_list).tolist()  # 导出结果（array类型→list类型）
    # print(np.array(tmp_data).shape) # (23762, 4)
    # tmp_data = np.array(tmp_data).T.tolist()
    return stdsc, tmp_data


def categorical_feature_conversion(df_cat):
    """
    # 将分类特征转换为数值型
    # 1、LightGBM可以直接用类别特征进行训练，不必预先进行独热编码
    #    参数设置categorical_feature来指定数据中的类别特征列(通过索引标记)
    #    避免将他们的大小关系作为特征进行学习；
    # 2、引入独热编码（不推荐）；
    """
    label_encoders = []
    categorical_features = []
    le = preprocessing.LabelEncoder()
    for i in range(0, df_cat.shape[1]):
        column_name = df_cat.columns[i]
        column_type = df_cat[column_name].dtypes
        # df_cat[column_name] = df_cat[column_name].astype('category')

        # if column_type == "object":
        df_cat[column_name] = list(map(lambda x: str(x), df_cat[column_name]))
        le.fit(df_cat[column_name])

        # 类别转换mapping需要保存用于predict
        mapping = dict(zip(le.classes_, range(1, len(le.classes_) + 1)))
        label_encoders.append(mapping.copy())

        feature_classes = list(le.classes_)
        encoded_feature = le.transform(df_cat[column_name]) + 1 # Transform Categories Into Integers 从1开始
        # print(type(encoded_feature))    # <class 'numpy.ndarray'>
        # print(le.inverse_transform([0, 0, 1, 2])) # Transform Integers Into Categories
        df_cat[column_name] = pd.DataFrame(encoded_feature)

        categorical_features.append(column_name)

    # df_cat = df_cat.drop_duplicates() # 去重
    return label_encoders, df_cat


# 读取数据
df = ut.DataPersistence.readExcel(setting.data_bucketB + setting.ltr_data_excel, tolist=False)

# 拿到MinMaxScaler和缩放后的特征
col_names = df.columns.values.tolist()  # header
name_scale = list(set(feat_name_scale).intersection(set(col_names)))  # 特征交集
print("对有意义的特征值放缩...", name_scale)    # ['time_diff']
if name_scale:
    feat_list = [df[name] for name in name_scale]   # 取特征列
    feat_list = [list(x) for x in zip(*feat_list)]  # 转换为特征行
    stdsc, feat_list = feature_scaling(feat_list)   # 执行特征缩放
    feat_list = [list(map(lambda x: round(x, 5), feat)) for feat in feat_list]  # 小数点保留5位
    # print(np.array(feat_list).shape)
else:
    stdsc, feat_list = None, []

# 拿到 other 特征
if "searchOrder" in feat_name_other:
    df["searchOrder"] = [x/max_page_size for x in df["searchOrder"]]
feat_orther_list = [df[name] for name in feat_name_other]   # 取特征列
feat_orther_list = [list(x) for x in zip(*feat_orther_list)]  # 转换为特征行
if feat_list:
    for i in range(len(feat_orther_list)):
        line = feat_orther_list[i]
        feat_list[i].extend(line)   # 特征合并
    # print(np.array(feat_list).shape)
else:
    feat_list = feat_orther_list[:]

# 拿到LabelEncoder和转换后的类别特征
name_categ = list(set(feat_name_categ).intersection(set(col_names)))  # 特征交集
if name_categ:
    print("分类特征转换...", name_categ)
    feat_categ = [df[name] for name in name_categ]   # 取特征列
    label_encoders, feat_categ = categorical_feature_conversion(pd.concat(
        feat_categ,
        axis=1,  # axis: {0/’index’, 1/’columns’}, default 0
    ))   # 转换类别特征为递增id
    
    nb_categorical_feature = -feat_categ.shape[1]  
    print("倒数{}个特征是类别特征".format(nb_categorical_feature))  # -3
    
    assert np.array(feat_list).shape[0]==feat_categ.shape[0]
    for i in range(feat_categ.shape[0]):
        line = feat_categ.iloc[i].values  # 特征行
        # line = list(map(lambda x: str(x + 1), line))  # 避免0
        feat_list[i].extend(line)   # 特征合并
        
    feat_categ = None
else:
    print("分类特征为空！")
    label_encoders = []
    nb_categorical_feature = 0
print("组合后特征shape=", np.array(feat_list).shape)  # (49069, 7)


# 所有可用特征的集合 (按照feature_name_all的顺序依次写入特征)
feature_name_all = name_scale + feat_name_other + name_categ
assert len(feature_name_all)==np.array(feat_list).shape[1]
print("所有可用特征: ", feature_name_all)
with codecs.open(setting.data_bucketB + setting.feature_all_file, 'w', 'utf-8') as f:
    f.write('\n'.join(feature_name_all))

qid, prev = 0, ""
duplicate_items = set() # 避免重复条目
rows = len(feat_list)
print("构造 libsvm 格式的数据")
with codecs.open(setting.data_bucketB + setting.ltr_data_txt, "w", "utf-8") as f:
    for i in range(rows):
        feat_idx = 1    # 每行中每个特征的序号-递增
        query = df.loc[i, "query"]
        # 控制qid递增
        if query != prev: 
            qid += 1
            prev = query
            duplicate_items = set()  # 新的qid重置
        # 一行特征值构造
        line = "{} qid:{} ".format(df.loc[i, "label"], qid)
        line += " ".join(["{}:{}".format(idx+1, feat) for idx, feat in enumerate(feat_list[i])])
        line += " #{} {} {} {}".format(
            repr(re.sub(" ", "", query)),
            repr(re.sub(" ", "", df.loc[i, "primaryQues"])),
            repr(re.sub(" ", "", df.loc[i, "botCode"])),
            repr(re.sub(" ", "-", df.loc[i, "opeTime"])),
        )
        
        # # 组装(一行)数据
        # for k, feat_name in enumerate(feature_name_all):
        #     if feat_name in name_scale:  # 缩放特征→round
        #         # feat_scale_idx = name_scale.index(feat_name)
        #         line += " {}:{}".format(feat_idx, round(feat_list[i][k], 5))
        #     elif feat_name in name_categ:  # id类别特征
        #         line += " {}:{}".format(feat_idx, feat_list[i][k])
        #     elif feat_name == "searchOrder":
        #         line += " {}:{}".format(feat_idx, (max_page_size - df.loc[i, feat_name]) / max_page_size)
        #     else:
        #         line += " {}:{}".format(feat_idx, df.loc[i, feat_name])
        #     feat_idx += 1
        # line += " #{} {} {} {}".format(
        #     repr(re.sub(" ", "", query)),
        #     repr(re.sub(" ", "", df.loc[i, "primaryQues"])),
        #     repr(re.sub(" ", "", df.loc[i, "botCode"])),
        #     repr(re.sub(" ", "-", df.loc[i, "opeTime"])),
        # )

        # # 组装(一行)数据
        # item = (
        #     "{} qid:{} 1:{} 2:{} 3:{} 4:{} "
        #     + ":{} ".join([str(i + 5) for i in range(5)])
        #     + ":{} #{} {} {} {}"
        # )
        # item = item.format(
        #     df.loc[i, "label"],
        #     qid,
        #     df.loc[i, "score"],  # if df.loc[i, "score"]!=0.00000 else 0.00001,
        #     df.loc[i, "score_tfidf"],
        #     df.loc[i, "score_bow"],
        #     round(feat_list[i][0], 5),  # score_bm25,
        #     df.loc[i, "score_doc2vec"],
        #     df.loc[i, "score_bert"],
        #     (max_page_size - df.loc[i, "searchOrder"]) / max_page_size,  # searchOrder
        #     round(feat_list[i][1], 5),  # 时间差
        #     round(df.loc[i, "word_occurrence"], 5),  # 编辑距离ratio
        #     # feat_list[i][4],     # title_id
        #     # feat_list[i][5],     # answer_id
        #     # feat_list[i][6],     # category_id
        #     # feat_list[i][7],     # baseCode
        #     # feat_list[i][8],     # projectCode
        #     repr(re.sub(" ", "", df.loc[i, "query"])),
        #     repr(re.sub(" ", "", df.loc[i, "title"])),
        #     repr(re.sub(" ", "", df.loc[i, "botCode"])),
        #     repr(re.sub(" ", "-", df.loc[i, "opeTime"])),
        # )
        
        # 避免重复条目写入文件
        if line.split("#")[0] not in duplicate_items:
            duplicate_items.add(line.split("#")[0])
            f.write(line)
            f.write("\n")


print("划分训练/验证/测试集")
with codecs.open(setting.data_bucketB + setting.ltr_data_txt, "r", "utf-8") as f:
    lines = f.readlines()
totals = len(lines)
a = int(totals * 0.8)
b = int(totals * 0.9)
with codecs.open(setting.data_bucketB + setting.ltr_train, "w", "utf-8") as f:
    for i in range(0, b):
        f.write(lines[i])
with codecs.open(setting.data_bucketB + setting.ltr_valid, "w", "utf-8") as f:
    for i in range(b, totals):
        f.write(lines[i])
with codecs.open(setting.data_bucketB + setting.ltr_test, "w", "utf-8") as f:
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
        setting.data_bucketB + setting.ltr_train,
        setting.data_bucketB + setting.lxh_train,
        setting.data_bucketB + setting.lxh_train_group,
    ]
)
returncode = loader.wait()  # 阻塞直至子进程完成
loader = subprocess.Popen(
    [
        "python",
        "trans_data.py",
        setting.data_bucketB + setting.ltr_valid,
        setting.data_bucketB + setting.lxh_valid,
        setting.data_bucketB + setting.lxh_valid_group,
    ]
)
returncode = loader.wait()  # 阻塞直至子进程完成
loader = subprocess.Popen(
    [
        "python",
        "trans_data.py",
        setting.data_bucketB + setting.ltr_test,
        setting.data_bucketB + setting.lxh_test,
        setting.data_bucketB + setting.lxh_test_group,
    ]
)
returncode = loader.wait()  # 阻塞直至子进程完成


# 保存必要的中间变量(用于预测)
joblib.dump(nb_categorical_feature, setting.data_bucketB + setting.cat_feat_file)  # 倒数几个特征是属于类别特征
joblib.dump(stdsc, setting.data_bucketB + setting.stdsc_file)  # 数据放缩字典
joblib.dump(label_encoders, setting.data_bucketB + setting.label_encoder_file)  # 保存特征转换的字典
