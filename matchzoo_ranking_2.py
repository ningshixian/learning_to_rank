import os
import codecs
import csv
import re
import typing
# from matchzoo import metrics
import numpy as np
from pathlib import Path
from numpy.lib.function_base import average
from pandas.core.algorithms import mode
from sklearn import preprocessing  # 用于正则化
import matchzoo as mz
from scipy.stats import spearmanr, pearsonr, kendalltau
import pandas as pd
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

import setting
from ndcg import validate


def build_data(path):
    data = {"sentence1": [], "sentence2": [], "label": []}
    with codecs.open(path, "r", "utf-8") as fp:
        for line in fp:
            if not line:
                break
            if "#" in line:
                data["label"].append(int(line.strip().split(" ")[0]))
                comment = line[line.index("#") + 1 :].strip("\n").split(" ")
                data["sentence1"].append(re.sub("'", "", comment[0]))
                data["sentence2"].append(re.sub("'", "", comment[1]))
    return data


# data = build_data(setting.ltr_data_txt)
# avg = average([len(x) for x in data["sentence1"]])
# avg2 = average([len(x) for x in data["sentence2"]])
# print(avg)  # 5.468454717255415
# print(avg2) # 12.335513882831806


def read_data(path):
    table = build_data(path)
    df = pd.DataFrame(
        {
            "text_left": table["sentence1"],
            "text_right": table["sentence2"],
            "label": table["label"],
        }
    )
    # 将读取之后的DataFrame类型转换成DataPack类型
    return mz.pack(df)


def load_data(
    stage: str = "train",
    task: str = "ranking",
    filtered: bool = False,
    return_classes: bool = False,
) -> typing.Union[mz.DataPack, tuple]:
    if stage not in ("train", "dev", "test"):
        raise ValueError(
            f"{stage} is not a valid stage."
            f"Must be one of `train`, `dev`, and `test`."
        )

    data_pack = read_data(setting.ltr_data_txt)
    # 定义任务类型和metrics
    if task == "ranking":
        task = mz.tasks.Ranking()
    if task == "classification":
        task = mz.tasks.Classification()

    if isinstance(task, mz.tasks.Ranking):
        return data_pack
    elif isinstance(task, mz.tasks.Classification):
        data_pack.one_hot_encode_label(task.num_classes, inplace=True)
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(
            f"{task} is not a valid task."
            f"Must be one of `Ranking` and `Classification`."
        )


# 1. 指定任务类型
task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=10))
# task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision(),
]

# 2. 准备数据
# 2.1 封装成DataPack (27516, 5)
data_pack = load_data(stage='train', task=task)
print(data_pack.frame().shape)

# 划分训练集和测试集
num_train = int(len(data_pack) * 0.8)
data_pack.shuffle(inplace=True)   # 打乱
train_raw = data_pack[:num_train]
test_raw = data_pack[num_train:]
# print(train_raw.left.head())
# print(train_raw.right.head())
# print(train_raw.relation.head())
print(train_raw.frame().head())

# # 读取词向量并构建词向量矩阵
# # path_vec = "D:/wordEmbedding/Tencent_AILab_ChineseEmbedding_small.txt"
# path_vec = "D:/wordEmbedding/word_vector.txt"
# emb = mz.embedding.load_from_file(path_vec, mode="bin")

# # 列出所有预处理类，需对中文处理进行适配
# print(mz.preprocessors.list_available())

# matchzoo仅支持英文形式(nltk分词工具以空格分词)
# 默认处理步骤: Tokenize → Lowercase → PuncRemoval → 对text_left/text_right的固定长度填充 → 字符过滤
model_class = mz.models.MVLSTM
preprocessor_class = mz.models.MVLSTM.get_default_preprocessor()
preprocessor_class._units = [
    mz.preprocessors.units.tokenize.ChineseTokenize(),  # 采用结巴分词
    mz.preprocessors.units.punc_removal.PuncRemoval(),
]

# """ ================matchzoo自带示例================== """
# train_raw = mz.datasets.wiki_qa.load_data("train", task=task)
# dev_raw = mz.datasets.wiki_qa.load_data("dev", task=task)
# test_raw = mz.datasets.wiki_qa.load_data("test", task=task)
# print(train_raw.left.head())
# print(train_raw.frame().head())
# model_class = mz.models.MVLSTM
# preprocessor_class = mz.models.MVLSTM.get_default_preprocessor()
# # emb = mz.datasets.embeddings.load_glove_embedding(dimension=300)
# """ ================matchzoo自带示例================== """

# Shortcut 
print(mz.auto.Preparer.get_default_config())
model, preprocessor, data_generator_builder, embedding_matrix = mz.auto.prepare(
    task=task,
    model_class=model_class,
    preprocessor=preprocessor_class,  #
    data_pack=train_raw,
    # embedding=emb,  #
)
# print("embedding_matrix: \n", type(embedding_matrix), "\n", embedding_matrix[:2])


# 数据预处理
print(preprocessor.context)
train_processed = preprocessor.transform(train_raw, verbose=0)  # , groupby
test_processed = preprocessor.transform(test_raw, verbose=0)
# print(train_processed.frame().shape)    # (22012, 7)

# vocab_unit = preprocessor.context["vocab_unit"]  # 此部分是为了显示处理过程
# sequence = train_processed.left.loc["L-2"]["text_left"]
# print("Transformed Indices:", sequence)
# print(
#     "Transformed Indices Meaning:",
#     "/".join([vocab_unit.state["index_term"][i] for i in sequence]),
# )

# 定义模型和参数
print(model.params.to_frame()[['Name', 'Description', 'Value']])  # 展示模型中可调参数
# model.params["input_shapes"] = [(30,), (30,)]  # [(10,), (40,)]
# model.params["mlp_num_units"] = 128  # 隐层大小
# model.params["optimizer"] = "adam"  # 直接调整参数    "adadelta"
# model.params["embedding_trainable"] = True  # 直接调整参数
# model.params["embedding_output_dim"] = 50  # 直接调整参数
# model.params["mlp_activation_func"] = "relu"  # 激活函数
model.params["dropout_rate"] = 0.5  # dropout
model.guess_and_fill_missing_params(verbose=0)
model.params.completed()
model.build()
model.backend.summary()
model.compile()


### 训练, 评估, 预测
x, y = train_processed.unpack()
test_x, test_y = test_processed.unpack()
train_gen = data_generator_builder.build(train_processed, mode='pair')
test_gen = data_generator_builder.build(test_processed, mode='point')

# # 输出数据生成器的数据(无1标签的group会被过滤掉)
# x, y = train_gen[0]
# print(y[:20])
# for key, value in sorted(x.items()):
#     print(key, str(value)[:50])
# x, y = test_gen[0]
# print(y)
# for key, value in sorted(x.items()):
#     print(key, str(value)[:50])
# exit()

evaluate = mz.callbacks.EvaluateAllMetrics(model, x=test_x, y=test_y)   # , model_save_path='model/my-model'
model.fit_generator(
    train_gen, epochs=2, callbacks=[evaluate], workers=5, use_multiprocessing=False
)
eva = model.evaluate_generator(test_gen)


### ndcg计算
left_id = test_x["id_left"]
right_id = test_x["id_right"]
y_pred = model.predict(test_x)
y_pred = [x[0] for x in y_pred]
test_y = np.array([x[0] for x in test_y])
average_ndcg, _ = validate(left_id, test_y, y_pred, 200)  # 所有qid的平均ndcg
print(
    "all qid average ndcg: ", average_ndcg)
print("斯皮尔曼相关性系数:", spearmanr(test_y, y_pred))
# 每个项目的相关性得分与预测得分之间的相关性
print("皮尔森相关性系数:", pearsonr(test_y, y_pred))
print("job done!")

## 保存模型
import shutil
if os.path.exists(setting.save_path):
    shutil.rmtree(setting.save_path)
model.save(setting.save_path)
preprocessor.save(setting.preprocessor_path)






# 加载MV-LSTM模型
# import matchzoo as mz
# mvlstm_model = mz.load_model(setting.save_path)
# preprocessor = mz.load_preprocessor(setting.preprocessor_path)

# data = {"sentence1": [], "sentence2": [], "label": []}

# data["sentence1"] = ["图纸", "图纸", "图纸"]
# data["sentence2"] = ["图纸不能上传", "审图系统看不到图纸", "审图上传图纸要求"]
# data["label"] = [0] * len(data["sentence1"])

# df = pd.DataFrame(
#     {
#         "text_left": data["sentence1"],
#         "text_right": data["sentence2"],
#         "label": data["label"],
#     }
# )
# df = mz.pack(df)
# test_processed = preprocessor.transform(df, verbose=0)
# test_x, test_y = test_processed.unpack()
# p = model.predict(test_x)
# print(p)


# @app.post("/resort")
# @log_filter
# def main(request: Request, item: Item):
#     json_data = item.dict()
#     query = json_data.get("query")
#     result_ori = json_data.get("result")
#     knowledge_list = json_data.get("knowledgeList")

#     result = [x.copy() for x in result_ori]  # [{}]的深拷贝!!!
#     if not result:  # 如果结果为空
#         return json_data

#     # 数据清洗
#     query = ''.join([x for x in query if x.isalnum()])
#     for item in result:
#         item['primary'] = ''.join([x for x in item['primary'] if x.isalnum()])

#     # MV-LSTM model
#     data = {"sentence1": [], "sentence2": [], "label": []}
#     data["sentence1"] = [query] * int(json_data.get("totals"))
#     data["sentence2"] = [x['primary'] for x in result]
#     data["label"] = [0] * len(data["sentence1"])
#     df = pd.DataFrame(
#         {
#             "text_left": data["sentence1"],
#             "text_right": data["sentence2"],
#             "label": data["label"],
#         }
#     )
#     df = mz.pack(df)
#     test_processed = preprocessor.transform(df, verbose=0)
#     test_x, test_y = test_processed.unpack()
#     logger.info(test_x)
#     preds_test = mvlstm_model.predict(test_x)
#     preds_test = [x[0] for x in preds_test]

#     # 针对query=primary的情况，直接将其概率置为1！！！
#     for i, x in enumerate(preds_test):
#         preds_test[i] = x if query != result[i]["primary"] else 1.0
#     # print('得分：', preds_test)

#     # 根据group切分预测概率并排序，最终得到每个query的doc重排结果
#     # the predicted results are the relevant scores.
#     # And you can use the relevant scores to get the correct ranking order results by yourself, and qid is needed in this step.
#     rank = resort(preds_test)
#     temp = [{}] * len(result)
#     knowledgeList = [''] * len(result)
#     for i, x in enumerate(rank):
#         result_ori[i]["searchOrder"] = x + 1
#         temp[x] = result_ori[i]
#         knowledgeList[x] = knowledge_list[i]

#     return {
#         "result": temp,
#         "msg": json_data.get("msg"),
#         "total": json_data.get("total"),
#         "totals": json_data.get("totals"),
#         # "maxScore": None,
#         "knowledgeList": knowledgeList,
#         "experiment": "gdbt+lambdarank",
#     }
