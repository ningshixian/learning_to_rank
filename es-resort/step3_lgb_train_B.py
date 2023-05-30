import os
import pickle as pkl
from datetime import datetime
import sys
import codecs
import json
import matplotlib.pylab as plt
import lightgbm as lgb
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
from sklearn import datasets as ds
import pandas as pd
import numpy as np
import base64
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import spearmanr, pearsonr, kendalltau
import requests
from helpers import read_dataset, trans_to_dataframe
from ndcg import validate
import setting

"""
2020.11.24
train: 0.9758345390177736
test : 0.9441163463266602
斯皮尔曼相关性系数: SpearmanrResult(correlation=0.143518, pvalue=2.2431291056466033e-17)
皮尔森相关性系数: (0.177185, 8.751817472277112e-26)

2020.12.01
train: 0.987129422537058
test : 0.9651278000625605
斯皮尔曼相关性系数: SpearmanrResult(correlation=0.17567380205819122, pvalue=2.2495600074945146e-25)
皮尔森相关性系数: (0.2730789397509474, 3.2952225375963606e-60)
"""


K = 200  # NDCG@K
NUM_BOOST_ROUND = 800  # 迭代的次数 n_estimators/num_iterations/num_round/num_boost_round
EARLY_STOPPING_ROUNDS = 50
STATIC_PARAMS = {
    "task": "train",  # task type, support train and predict
    "objective": "lambdarank",  # 排序任务(目标函数)
    "boosting_type": "gbdt",  # 基学习器 gbrt dart
    "metric": "ndcg",  # 度量的指标(评估函数) MAP@3
    "ndcg_at": [1, 3, 5, 10],
    "max_position": 5,  # @NDCG 位置优化 5
    "metric_freq": 100,  # 每隔多少次输出一次度量结果
    "train_metric": True,  # 训练时就输出度量结果 True
    "tree_learner": "serial",  # 用于并行学习
    "num_threads": 2,  # 线程数，可以限制模型训练时CPU的占用率！！！http://ponder.work/2020/01/25/lightgbm-hang-in-multi-thread/
    # "device":"gpu"
    # 'is_unbalance': 'true',  #当训练数据的正负样本相差悬殊时，可以将这个属性设为true, 自动给少的样本赋予更高的权重
    "verbose": -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
N_TRIALS = 200  # 随机搜索的次数trail


# def split_data_from_keyword(data_read, data_group, data_feats):
#     """
#     利用pandas
#     转为lightgbm需要的格式进行保存
#     :param data_read:
#     :param data_save:
#     :return:
#     """
#     with open(data_group, "w", encoding="utf-8") as group_path:
#         with open(data_feats, "w", encoding="utf-8") as feats_path:
#             dataframe = pd.read_csv(
#                 data_read, sep=" ", header=None, encoding="utf-8", engine="python"
#             )
#             current_keyword = ""
#             current_data = []
#             group_size = 0
#             for _, row in dataframe.iterrows():
#                 feats_line = [str(row[0])]
#                 for i in range(2, len(dataframe.columns) - 1):
#                     feats_line.append(str(row[i]))
#                 if current_keyword == "":
#                     current_keyword = row[1]
#                 if row[1] == current_keyword:
#                     current_data.append(feats_line)
#                     group_size += 1
#                 else:
#                     for line in current_data:
#                         feats_path.write(" ".join(line))
#                         feats_path.write("\n")
#                     group_path.write(str(group_size) + "\n")

#                     group_size = 1
#                     current_data = []
#                     current_keyword = row[1]
#                     current_data.append(feats_line)

#             for line in current_data:
#                 feats_path.write(" ".join(line))
#                 feats_path.write("\n")
#             group_path.write(str(group_size) + "\n")


# def save_data(group_data, output_feature, output_group):
#     """
#     group与features分别进行保存
#     """
#     if len(group_data) == 0:
#         return
#     output_group.write(str(len(group_data)) + "\n")
#     for data in group_data:
#         # 只包含非零特征
#         # feats = [p for p in data[2:] if float(p.split(":")[1]) != 0.0]
#         feats = [p for p in data[2:]]
#         output_feature.write(
#             data[0] + " " + " ".join(feats) + "\n"
#         )  # data[0] => level ; data[2:] => feats


# def process_data_format(test_path, test_feats, test_group):
#     """
#      转为lightgbm需要的格式进行保存
#      """
#     with open(test_path, "r", encoding="utf-8") as f_read:
#         with open(test_feats, "w", encoding="utf-8") as output_feature:
#             with open(test_group, "w", encoding="utf-8") as output_group:
#                 group_data = []
#                 group = ""
#                 for line in f_read:
#                     if "#" in line:
#                         line = line[: line.index("#")]
#                     splits = line.strip().split()
#                     if splits[1] != group:  # qid => splits[1]
#                         save_data(group_data, output_feature, output_group)
#                         group_data = []
#                         group = splits[1]
#                     group_data.append(splits)
#                 save_data(group_data, output_feature, output_group)


def load_data(feats, group):
    """
    加载数据
    分别加载feature,label,query
    """
    x_train, y_train = ds.load_svmlight_file(feats)
    q_train = np.loadtxt(group)
    print("x: ", x_train.shape)  # (31118, 7)
    print("y: ", y_train.shape)  # (31118,)
    print("group: ", sum(q_train))  # 31118.0

    # x_train = trans_to_dataframe(x_train, cate_cols)
    # print("dtypes: \n", x_train.dtypes)
    return x_train, y_train, q_train


def load_data_from_raw(raw_data):
    with open(raw_data, "r", encoding="utf-8") as testfile:
        test_X, test_y, test_qids, comments = read_dataset(testfile)
        # test_X = trans_to_dataframe(test_X, cate_cols)
    return test_X, test_y, test_qids, comments


def train(t_number, params):
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,  # 控制迭代次数
        valid_sets=[dev_data],
        categorical_feature=cate_cols,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=-1,  # 不显示每次迭代在验证集的度量结果 100
    )
    
    # # Save a trained model to a file.
    # gbm.save_model(setting.model_root + '{}.mod'.format(t_number), num_iteration=gbm.best_iteration)  # 会保存最佳迭代结果:
    with open(setting.model_bucketB + setting.gbdt_model.format(t_number), "wb") as f:
        # f.seek(0)
        pkl.dump(gbm, f)

    # 保存当前trial的最佳迭代次数
    with codecs.open(setting.model_bucketB + setting.best_iteration.format(t_number), "w", "utf-8") as f:
        f.write(str(gbm.best_iteration))

    # test_predict = gbm.predict(test_X, num_iteration=gbm.best_iteration, categorical_feature=cate_cols)
    # average_ndcg, _ = validate(test_qids, test_y, test_predict, K)    # 所有qid的平均ndcg
    # # average_ndcg, _ = validate(q_test, test_y, test_predict, K)    # 所有group的平均ndcg
    # return average_ndcg

    score = gbm.best_score["valid_0"]["ndcg@5"]
    return score


def objective(trial):
    """
    模型的训练和保存
    https://github.com/optuna/optuna/blob/master/examples/lightgbm_simple.py
    https://zhuanlan.zhihu.com/p/138521995
    """
    params = {
        # "learning_rate": trial.suggest_loguniform(
        #     "learning_rate", 0.01, 0.5
        # ),  # 学习率 [0.1, 0.01, 0.001, 0.005]
        "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.1),  # 学习率
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "num_leaves": trial.suggest_int(
            "num_leaves", 5, 512, 5
        ),  # 叶子数 取值应 <= 2 ^（max_depth） 从1～3范围内的int里选 ≈max_depth
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", 1, 256, 10
        ),  # 一个叶子节点上包含的最少样本数量
        "max_bin": trial.suggest_int("max_bin", 5, 256, 10),  # 一个整数，表示最大的桶的数量。默认值为 255
        "feature_fraction": trial.suggest_categorical("feature_fraction", [0.5, 0.6,0.7,0.8,0.9,1.0]),
        "bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.5, 0.6,0.7,0.8,0.9,1.0]),
        'bagging_freq': trial.suggest_int("bagging_freq", 0, 128, 10),
        "lambda_l1": trial.suggest_categorical("lambda_l1", [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]),
        "lambda_l2": trial.suggest_categorical("lambda_l2", [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]),
        'min_split_gain':trial.suggest_discrete_uniform("min_split_gain", 0.0, 1.0, 0.1),
        
        "subsample": trial.suggest_discrete_uniform("subsample", 0.1, 1.0, 0.1),
        "min_child_samples": trial.suggest_categorical(
            "min_child_samples", [100, 200, 300, 400, 500]
        ),
        "min_child_weight": trial.suggest_categorical(
            "min_child_weight", [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        ),
        "num_iterations": trial.suggest_int("num_iterations", 200, 1000, 100),  # 迭代次数？即num_trees（生成的决策树的数量）
    }
    all_params = {**params, **STATIC_PARAMS}
    return train(trial.number, all_params)


def predict(x_test, comments, model_input_path):
    """
    预测得分并排序
    """
    # 加载model
    # gbm = lgb.Booster(model_file=model_input_path)  
    with open(model_input_path, "rb") as fin:
        gbm = pkl.load(fin)

    ypred = gbm.predict(
        x_test, num_iteration=best_iteration, categorical_feature=cate_cols
    )
    predicted_sorted_indexes = np.argsort(ypred)[::-1]  # 返回从大到小的索引
    t_results = comments[predicted_sorted_indexes]  # 返回对应的comments,从大到小的排序
    return t_results


def test_data_ndcg(model_path, test_path):
    """
    评估测试数据的ndcg
    """
    # with open(test_path, 'r', encoding='utf-8') as testfile:
    #     test_X, test_y, test_qids, comments = read_dataset(testfile)
    #     test_X = trans_to_dataframe(test_X, cate_cols)
    # q_test = np.loadtxt(setting.lxh_test_group)

    # gbm = lgb.Booster(model_file=model_path)
    with open(model_path, "rb") as fin:
        gbm = pkl.load(fin)
    test_predict = gbm.predict(
        test_X, num_iteration=best_iteration, categorical_feature=cate_cols
    )

    average_ndcg, _ = validate(test_qids, test_y, test_predict, K)  # 所有qid的平均ndcg
    # average_ndcg, _ = validate(q_test, test_y, test_predict, K)  # 所有group的平均ndcg

    print(
        "all qid average ndcg: ", average_ndcg
    )  # 测试ES：0.8834129447812231；生产ES：0.6856617379112661
    print("斯皮尔曼相关性系数:", spearmanr(test_y, test_predict))
    # 每个项目的相关性得分与预测得分之间的相关性
    print("皮尔森相关性系数:", pearsonr(test_y, test_predict))
    print("job done!")


def test_data_excel(model_path, test_path):
    """将测试集的预测结果写入Excel"""
    # with open(test_path, "r", encoding="utf-8") as testfile:
    #     test_X, test_y, test_qids, comments = read_dataset(testfile)
    #     test_X = trans_to_dataframe(test_X, cate_cols)
    # q_test = np.loadtxt(setting.lxh_test_group)

    # gbm = lgb.Booster(model_file=model_path)
    with open(model_path, "rb") as fin:
        gbm = pkl.load(fin)
    test_predict = gbm.predict(
        test_X, num_iteration=best_iteration, categorical_feature=cate_cols
    )

    start = 0
    predict_order = []
    for group in q_test:
        end = start + int(group)
        predicted_sorted_indexes = np.argsort(test_predict[start:end])[::-1]  # 从大到小的索引
        predict_order.extend(predicted_sorted_indexes + 1)
        start = end

    # print(test_X.columns.values.tolist())
    res = {
        "label": test_y,
        "query": [x.split()[0] for x in comments],
        "doc": [x.split()[1] for x in comments],
        "order": list(test_X[3]),
        "predict": predict_order,
    }
    df = pd.DataFrame(res)
    df.to_excel(setting.data_bucketB + setting.ltr_output_excel, index=False)


def plot_print_feature_importance(model_path):
    """
    打印特征的重要度
    """
    # feats_dict = {}
    # for feat_index in range(46):
    #     col = "Column_" + str(feat_index)
    #     feats_dict[col] = "feat" + str(feat_index) + "name"

    if not os.path.exists(model_path):
        print("file no exists! {}".format(model_path))
        sys.exit(0)

    # gbm = lgb.Booster(model_file=model_path)
    with open(model_path, "rb") as fin:
        gbm = pkl.load(fin)

    # 打印和保存特征重要度
    importances = gbm.feature_importance(importance_type="split")
    feature_names = gbm.feature_name()
    print("feature_names: ", feature_names)
    
    # # 画图
    # plt.figure(figsize=(12,6))
    # lgb.plot_importance(gbm, max_num_features=30)
    # plt.title("Featurertances")
    # plt.savefig(setting.model_bucketB + "feat_importance.png")

    # 特征列id
    feats_dict = {}
    for i in range(len(feature_names)):
        feats_dict[feature_names[i]] = i

    sum = 0.0
    for value in importances:
        sum += value

    # 导出特征重要性
    f = open(setting.model_bucketB + setting.feature_txt, "w+")
    for feature_name, importance in zip(feature_names, importances):
        if importance != 0:
            # feat_id = int(feature_name) + 1
            feat_id = int(feats_dict[feature_name]) + 1
            line = "Column_{} : {} : {} : {}".format(
                feat_id, "%10s" % (feature_name), importance, importance / sum
            )
            print(line)
            f.write(line + "\n")
    f.close()


def get_leaf_index(data, model_path):
    """
    得到叶结点并进行one-hot编码
    """
    # gbm = lgb.Booster(model_file=model_path)
    # ypred = gbm.predict(data, pred_leaf=True)
    with open(model_path, "rb") as fin:
        gbm = pkl.load(fin)
    ypred = gbm.predict(
        data,
        num_iteration=best_iteration,
        categorical_feature=cate_cols,
        pred_leaf=True,
    )

    one_hot_encoder = OneHotEncoder()
    x_one_hot = one_hot_encoder.fit_transform(ypred)
    print(x_one_hot.shape)
    # print(x_one_hot.toarray())


def train_lr(model_path):
    """
    GBDT和LR的融合方案
    参考《推荐系统遇上深度学习(十)--GBDT+LR融合方案实战》
    """
    # gbm = lgb.Booster(model_file=model_path)
    with open(model_path, "rb") as fin:
        gbm = pkl.load(fin)
        num_leaf = gbm.params['num_leaves']
    
    y_pred = gbm.predict(x_train, num_iteration=best_iteration, categorical_feature=cate_cols, pred_leaf=True)
    print(np.array(y_pred).shape)
    print(y_pred[0])    
    
    print('Writing transformed training data')
    transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                        dtype=np.int64)  # N * num_tress * num_leafs
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
        transformed_training_matrix[i][temp] += 1
    
    y_pred = gbm.predict(test_X, num_iteration=best_iteration, categorical_feature=cate_cols, pred_leaf=True)
    print('Writing transformed testing data')
    transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
        transformed_testing_matrix[i][temp] += 1
    
    from sklearn.linear_model import LogisticRegression
    lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
    lm.fit(transformed_training_matrix,y_train)  # fitting the data
    y_pred_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label
    with open(setting.model_bucketB + 'LR.mod', "wb") as f:
        pkl.dump(lm, f)
        print("LR模型保存")
    
    NE = (-1) / len(y_pred_test) * sum(((1+test_y)/2 * np.log(y_pred_test[:,1]) +  (1-test_y)/2 * np.log(1 - y_pred_test[:,1])))
    print("Normalized Cross Entropy " + str(NE))



if __name__ == "__main__":

    with codecs.open(setting.data_bucketB + setting.feature_all_file, "r", "utf-8") as f:
        feature_name_all = f.read().split("\n")
        print("所有可用特征: ", feature_name_all)
    
    cate_cols = joblib.load(setting.data_bucketB + setting.cat_feat_file)
    cate_cols = list(range(cate_cols, 0))
    cate_cols = [feature_name_all[idx] for idx in cate_cols]
    print("{}是类别特征".format(cate_cols))

    raw_data_path = setting.data_bucketB + setting.ltr_train
    data_feats = setting.data_bucketB + setting.lxh_train
    data_group = setting.data_bucketB + setting.lxh_train_group

    # 获取最佳模型路径及最佳迭代次数
    best_model_path, best_iteration = None, None
    if os.path.exists(setting.model_bucketB + setting.best_trial):
        with codecs.open(setting.model_bucketB + setting.best_trial, "r", "utf-8") as f:
            best_trial_number = int(f.read().strip('\n'))
        best_model_path = setting.model_bucketB + "{}.mod".format(best_trial_number)
        with codecs.open(
            setting.model_bucketB + "{}.best_iteration.txt".format(best_trial_number),
            "r",
            "utf-8",
        ) as f:
            best_iteration = int(f.read().strip('\n'))

    # 加载训练数据
    x_train, y_train, q_train = load_data(data_feats, data_group)
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        group=q_train,
        feature_name=feature_name_all,
        categorical_feature=cate_cols,
        # free_raw_data=False,
    )
    # 加载验证数据
    x_dev, y_dev, q_dev = load_data(setting.data_bucketB + setting.lxh_valid, setting.data_bucketB + setting.lxh_valid_group)
    dev_data = lgb.Dataset(
        x_dev,
        label=y_dev,
        group=q_dev,
        feature_name=feature_name_all,
        categorical_feature=cate_cols,
        reference=train_data    # ???
    )
    # 加载测试数据
    raw_test_path = setting.data_bucketB + setting.ltr_test
    test_group = setting.data_bucketB + setting.lxh_test_group
    with open(raw_test_path, "r", encoding="utf-8") as testfile:
        test_X, test_y, test_qids, comments = read_dataset(testfile)
        # test_X = trans_to_dataframe(test_X, cate_cols)
    q_test = np.loadtxt(test_group)

    if len(sys.argv) < 2:
        print(
            "Usage: python main.py [-process | -train | -predict | -ndcg | -feature | -leaf]"
        )
        sys.exit(0)
    
    if sys.argv[1] == "-train":
        # # logs run scores and run parameters and plots the scores so far
        # neptune.init('ningshixian/sandbox', api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZjVkMTAzZTQtYWM2MC00ZDJhLWJkZGEtOGQzMDMyM2IxNTkwIn0=")
        # neptune.create_experiment(name='optuna sweep')
        # monitor = opt_utils.NeptuneMonitor()

        # 创建一个学习实例，因为objective返回的评价指标是ndcg，因此目标是最大化
        study = optuna.create_study(direction="maximize")
        # n_trials代表多少种参数组合，n_jobs是并行搜索的个数，-1代表使用所有的cpu核心
        study.optimize(
            objective, n_trials=N_TRIALS, n_jobs=1, gc_after_trial=True
        )  # , callbacks=[monitor]
        # opt_utils.log_study(study)

        print("最优超参: ", study.best_params)
        print("最优超参下，objective函数返回的值: ", study.best_value)
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # 保存最佳trial
        with codecs.open(setting.model_bucketB + setting.best_trial, "w", "utf-8") as f:
            f.write(str(study.best_trial.number))

        # 删除非最佳模型文件
        ls = os.listdir(setting.model_bucketB)
        for name in ls:
            if not name.split(".")[0] == str(study.best_trial.number) and (
                name.endswith(".best_iteration.txt") or name.endswith(".mod")
            ):
                os.remove(setting.model_bucketB + name)

    elif sys.argv[1] == "-push":  # 推送更新给测试146服务
        model_data = {}
        model_data["bucket"] = "B"
        # 模型相关的
        with codecs.open(setting.model_bucketB + setting.best_trial, "r", "utf-8") as f:
            model_data["trial"] = str(f.read())
        with codecs.open(
            setting.model_bucketB + setting.best_iteration.format(model_data["trial"]), "r", "utf-8"
        ) as f:
            model_data["iter"] = int(f.read())
        with open(setting.model_bucketB + setting.gbdt_model.format(model_data["trial"]), "rb") as fin:
            model_data["model64"] = base64.b64encode(fin.read()).decode("utf-8")
        # 特征相关的
        model_data["feature_name_all"] = list(feature_name_all)
        with open(setting.data_bucketB + setting.stdsc_file, "rb") as fin:
            model_data["stdsc"] = base64.b64encode(fin.read()).decode("utf-8")
        with open(setting.data_bucketB + setting.label_encoder_file, "rb") as fin:
            model_data["label_encoder"] = base64.b64encode(fin.read()).decode("utf-8")
        if "category_id" in feature_name_all:
            with open(setting.data_bucketB + setting.cat_feat_file, "rb") as fin:
                model_data["category_id"] = base64.b64encode(fin.read()).decode("utf-8")
        if "score_tfidf" in feature_name_all:
            with open(setting.data_bucketB + setting.tfidf_vec_file, "rb") as fin:
                model_data["score_tfidf"] = base64.b64encode(fin.read()).decode("utf-8")
        if "score_bow" in feature_name_all:
            with open(setting.data_bucketB + setting.bow_vec_file, "rb") as fin:
                model_data["score_bow"] = base64.b64encode(fin.read()).decode("utf-8")
        if "score_bm25" in feature_name_all:
            with open(setting.data_bucketB + setting.bm25_file, "rb") as fin:
                model_data["score_bm25"] = base64.b64encode(fin.read()).decode("utf-8")
        if "score_doc2vec" in feature_name_all:
            with open(setting.data_bucketB + setting.doc2vec_file, "rb") as fin:
                model_data["score_doc2vec"] = base64.b64encode(fin.read()).decode(
                    "utf-8"
                )
            with open(setting.data_bucketB + setting.sen2docvec_dict_file, "rb") as fin:
                model_data["sen2docvec_dict"] = base64.b64encode(fin.read()).decode(
                    "utf-8"
                )
        if "time_diff" in feature_name_all:
            with open(setting.data_bucketB + setting.time_feats_file, "rb") as fin:
                model_data["time_diff"] = base64.b64encode(fin.read()).decode("utf-8")
        
        if sys.argv[2] == "-test":
            ip_list = ["http://10.231.135.146:8098"]
        elif sys.argv[2] == "-pro":
            ip_list = ["http://10.231.9.138:8098", "http://10.231.9.139:8098", "http://10.231.25.132:8098", "http://10.231.25.133:8098"]
        elif sys.argv[2] == "-pre":
            ip_list = ["http://10.231.198.91:8098"]
        elif sys.argv[2] == "-local":
            ip_list = ["http://localhost:8098"]
        elif sys.argv[2] == "-154":
            ip_list = ["http://10.231.10.201:8098", "http://10.231.1.154:8098"]
        else:
            ip_list = []

        print("推送特征处理过程的中间对象：", model_data.keys())
        for url in ip_list:
            upload_res = requests.post(
                url + "/upload_model", json.dumps(model_data), timeout=240
            )
            print(upload_res)

    elif sys.argv[1] == "-predict":
        train_start = datetime.now()
        predict_data_path = setting.data_bucketB + setting.ltr_test  # 格式如ranklib中的数据格式
        # test_X, test_y, test_qids, comments = load_data_from_raw(predict_data_path)
        t_results = predict(test_X, comments, best_model_path)
        # print(t_results)
        train_end = datetime.now()
        consume_time = (train_end - train_start).seconds
        print("consume time : {}".format(consume_time))

    elif sys.argv[1] == "-ndcg":
        # ndcg
        test_path = setting.data_bucketB + setting.ltr_test  # 评估测试数据的平均ndcg
        test_data_ndcg(best_model_path, test_path)

    elif sys.argv[1] == "-test":  # 可视化结果
        test_path = setting.data_bucketB + setting.ltr_test
        test_data_excel(best_model_path, test_path)

    elif sys.argv[1] == "-feature":
        plot_print_feature_importance(best_model_path)

    elif sys.argv[1] == "-leaf":
        # 利用模型得到样本叶结点的one-hot表示
        raw_data = setting.data_bucketB + setting.ltr_test
        with open(raw_data, "r", encoding="utf-8") as testfile:
            test_X, test_y, test_qids, comments = read_dataset(testfile)
        get_leaf_index(test_X, best_model_path)
        
    elif sys.argv[1] == "-train_lr":
        train_lr(best_model_path)
