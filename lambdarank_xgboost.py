import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import DMatrix, train
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

"""xgb在进行ranking时，模式是pairwise，而pairwise主要的优化目标主要是排序对。
根据xgb官方文档，主要使用的是LambdaRank算法，详情参考LambdaRank算法。评估指标NDCG参考NDCG
https://www.jianshu.com/p/9caef967ec0a
"""


xgb_rank_params1 = {
    "booster": "gbtree",    # 学习器类型，可以是默认的gbtree, 也就是CART决策树，还可以是线性弱学习器gblinear以及DART。一般来说，我们使用gbtree就可以了，不需要调参。
    "eta": 0.1,  # learning_rate
    "gamma": 1.0,   # 参数的值越大，算法越保守
    "min_child_weight": 0.1,    # 决定最小叶子节点样本权重和,缺省值=1,避免过拟合. 值过高，会导致欠拟合
    "objective": "rank:pairwise",   # 问题类别以及对应的损失函数
    # "objective": "rank:ndcg",
    "eval_metric": "merror",
    "max_depth": 6,
    "num_boost_round": 10,
    "save_period": 0,
}

xgb_rank_params2 = {
    "bst:max_depth": 2,
    "bst:eta": 1,
    "silent": 1,
    "objective": "rank:pairwise",
    # "objective": "rank:ndcg",
    "nthread": 4,
    "eval_metric": "ndcg",
}


# generate training dataset
# 一共2组*每组3条，6条样本，特征维数是2
n_group = 2
n_choice = 3
dtrain = np.random.uniform(0, 100, [n_group * n_choice, 2])
# numpy.random.choice(a, size=None, replace=True, p=None)
dtarget = np.array(
    [np.random.choice([0, 1, 2], 3, False) for i in range(n_group)]
).flatten()
# n_group用于表示从前到后每组各自有多少样本，前提是样本中各组是连续的，[3，3]表示一共6条样本中前3条是第一组，后3条是第二组
dgroup = np.array([n_choice for i in range(n_group)]).flatten()
print(dtrain)
print(dtarget)

# 数据格式 : 第一列是 label，后面的特征，用 DMatrix
# concate Train data, very import here !
xgbTrain = DMatrix(dtrain, label=dtarget)
xgbTrain.set_group(dgroup)
# print(xgbTrain.num_row(), xgbTrain.num_col())

# generate eval data
dtrain_eval = np.random.uniform(0, 100, [n_group * n_choice, 2])
xgbTrain_eval = DMatrix(dtrain_eval, label=dtarget)
xgbTrain_eval.set_group(dgroup)
evallist = [(xgbTrain, "train"), (xgbTrain_eval, "eval")]

# train model
# xgb_rank_params1加上 evals 这个参数会报错，还没找到原因
# rankModel = train(xgb_rank_params1,xgbTrain,num_boost_round=10)
rankModel = train(xgb_rank_params2, xgbTrain, num_boost_round=20, evals=evallist)
# rankModel.save_model('xgb.model')

# test dataset
dtest = np.random.uniform(0, 100, [n_group * n_choice, 2])
dtestgroup = np.array([n_choice for i in range(n_group)]).flatten()
xgbTest = DMatrix(dtest)
xgbTest.set_group(dgroup)

# test
y_pred = rankModel.predict(xgbTest)
print(y_pred)
# y_true = np.array([0, 0, 1, 1, 0, 1])
# print(roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro'))

# # load model.
# bst = xgb.Booster()
# bst.load_model('xgb.model')