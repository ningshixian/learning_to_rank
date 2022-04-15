from scipy.stats import spearmanr, pearsonr, kendalltau
import lightgbm as lgb
import numpy as np
import codecs
import re 
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import setting
import helpers

"""
LambdaMART = Lambda(LambdaRank) + MART(GBDT)
lambda很巧妙的用来描述求解过程中的迭代方向和强度
学过adam优化方法的人可能会很熟悉，在梯度之前乘上一个系数，这个系数是用来影响梯度的方向的。

将同Query下有点击样本和无点击样本构造成一个样本Pair

代码示例：https://mlexplained.com/2019/05/27/learning-to-rank-explained-with-code/
lightgbm使用指南：http://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/lightgbm/chapters/lightgbm_usage.html
lightgbm参数说明：https://lanpeihui.top/2019/02/23/%E9%9B%86%E6%88%90%E7%AE%97%E6%B3%95%E4%B9%8BLightGBM/
"""

# 加载数据
"""函数参数：
load_svmlight_file(f, n_features=None, dtype=<type 'numpy.float64'>, multilabel=False,zero_based='auto', query_id=False)

参数介绍：
'f'为文件路径
'n_features'为feature的个数，默认None自动识别feature个数
'dtype'为数据集的数据类型（不是很了解），默认是np.float64
'multilable'为多标签，多标签数据的具体格式可以参照(这里)
"""

x_train, y_train = load_svmlight_file(setting.lxh_train)    # , dtype=np.float64
x_valid, y_valid = load_svmlight_file(setting.lxh_valid)
x_test, y_test = load_svmlight_file(setting.lxh_test)
# load_svmlight_file得到的特征矩阵X是一个SciPy CSR matrix → 转换成np.array格式
x_train, x_valid, x_test = x_train.toarray(), x_valid.toarray(), x_test.toarray()
print("训练集数据量：", x_train.shape[0])
print("训练集特征数：", x_train.shape[1])
# print(type(x_train))    # <class 'scipy.sparse.csr.csr_matrix'>
# print(type(y_train))    # <class 'numpy.ndarray'>
# print([item for item in x_train[:5]])
# exit()

q_train = np.loadtxt(setting.lxh_train_group)
q_valid = np.loadtxt(setting.lxh_valid_group)
q_test = np.loadtxt(setting.lxh_test_group)

categorical_features = joblib.load(setting.cat_feat_file)
print(categorical_features)

# # 合并成1个group进行训练
# q_train = [x_train.shape[0]]
# q_valid = [x_valid.shape[0]]
# q_test = [x_test.shape[0]]

# 模型训练
"""
- boosting_type：
    'gbdt'，传统的Gradient Boosting决策树。
    'dart'，带Dropouts的多个加法回归树。
    'goss'，Gradient-based One-Side Sampling。
    'rf'，随机森林
- objective： 默认值：LGBMRanker的'lambdarank'---pairwise!
- n_estimators=10: 拟合的树的数量，相当于训练轮数
"""

param_dist = {
    "max_depth": [25, 50, 75],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 300, 900, 1200],
    "n_estimators": [200, 300]
}
gbm = lgb.LGBMRanker(
    boosting_type="gbdt",
    objective="lambdarank",
    max_depth=3,
    max_bin=512,
    lambda_l1=7.360463659505314,
    lambda_l2=6.651184282278517,
    learning_rate=0.005,
    num_leaves=150,
    feature_fraction=0.7973782500135511,
    min_child_samples=9,
    min_data_in_leaf=80,
    n_estimators=400,
)
# grid_search = GridSearchCV(gbm, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
# grid_search.fit(x_train, y_train)
gbm.fit(
    x_train,
    y_train,
    group=q_train,
    eval_set=[(x_valid, y_valid)],
    eval_group=[q_valid],
    eval_at=[1, 3],
    early_stopping_rounds=200,
    eval_metric="ndcg",
    verbose=True,
    callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)],
    categorical_feature=categorical_features,  # 指明哪些特征的分类特征
)
print("\n")
print("object function: ", gbm.objective_)  # lambdarank
print("booster_: ", gbm.booster_)
# print("评估结果: ", gbm.evals_result_)
print("拟合模型的特征数: ", gbm.n_features_)  # 13
print("最佳拟合模型的分数: ", gbm.best_score_)

# # # 绘制特征的重要性
# ax = lgb.plot_importance(gbm, max_num_features=10)
# plt.show()
# plt.savefig("data/plot_importance.png")
# ax = lgb.plot_tree(gbm)
# plt.show()

# 导出特征重要性
importance = gbm.feature_importances_
names = [str(x) for x in range(1, gbm.n_features_ + 1)]
with open(setting.feature_txt, "w+") as file:
    for index, im in enumerate(importance):
        string = names[index] + ", " + str(im) + "\n"
        file.write(string)

# 模型存储
joblib.dump(gbm, setting.gbdt_model)
with codecs.open(setting.best_iteration, "w", "utf-8") as f:
    f.write(str(gbm.best_iteration_))
# 模型加载
gbm = joblib.load(setting.gbdt_model)

# 模型预测
# Return the predicted value for each sample.
# LightGBM will predict the relevant score, and you should rank the result by yourself.
preds_test = gbm.predict(
    x_test, num_iteration=gbm.best_iteration_
)  # 从最佳迭代中获得预测结果, raw_score=True

# for i in range(len(y_test)):
#     actual = y_test[i]
#     prediction = preds_test[i]
#     print("actual= {}, prediction= {}".format(actual, prediction))

# 模型评估
predictions_classes = []
for i in preds_test:
    if i<=0:
        label = 0
    elif i>0 and i<0.4:
        label = 1
    else:
        label = 2
    predictions_classes.append(label)

predictions_classes = np.array(predictions_classes)
accuracy = accuracy_score(predictions_classes, y_test)*100
print("测试集评估结果:")
print("准确度：{}%".format(accuracy))   # 98.40201850294366%
print("均方误差:", mean_squared_error(y_test, preds_test) ** 0.5)  # 0.5436660745760657
print("P/R/F: ", classification_report(y_true=y_test, y_pred=predictions_classes))

print("斯皮尔曼相关性系数:", spearmanr(y_test, preds_test)) # SpearmanrResult(correlation=0.9597422219752819, pvalue=0.0)
print("皮尔森相关性系数:", pearsonr(y_test, preds_test)) # (0.9826461314500071, 0.0)
# 统计学中的三大相关性系数：pearson, spearman, kendall
# https://www.cnblogs.com/yjd_hycf_space/p/11537153.html


def resort(probs):
    """根据概率排序，拿到排序后的索引"""
    new_ind = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)
    new_ind = sorted(enumerate(new_ind), key=lambda x: x[1])
    return [i for i, _ in new_ind]


print("拿到test数据集")
test_data = []
with codecs.open(setting.ltr_data_txt, 'r', 'utf-8') as f:
    for line in f:
        line = re.sub(r"[\'#]", "", line)
        splits = re.split(' ', line)
        test_data.append(splits)
        # if not len(splits)==16:
        #     print(line)
        #     print(splits)
start = int(len(test_data) * 0.9)
test_data = test_data[start:]
assert len(test_data) == len(preds_test)

start, end = 0, 0
for num in q_test:
    start, end = end, end + int(num)
    predict_ind = resort(preds_test[start:end])
    for i in range(int(num)):
        test_data[start:end][i] += [predict_ind[i] + 1]
test_data = np.array(test_data)

print("将预测结果写入Excel")

res = {
    "label": test_data[:, 0],
    "ctr": test_data[:, 4],
    "cosine-score": test_data[:, 2],
    "query": test_data[:, -5],
    "title": test_data[:, -4],
    "searchOrder": test_data[:, 3],
    "newOrder": test_data[:, -1],
}
df = pd.DataFrame(res)
df.to_excel(setting.ltr_output_excel, index=False)
