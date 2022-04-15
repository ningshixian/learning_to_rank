import re
import codecs
import matchzoo as mz
import pandas as pd

import setting


# # 定义任务，包含两种，一个是Ranking，一个是classification
# task = mz.tasks.Ranking()
# print(task)

# # 准备数据
# # train_raw是matchzoo中自定的数据格式	matchzoo.data_pack.data_pack.DataPack
# print(mz.datasets.list_available()) # ['toy', 'wiki_qa', 'embeddings', 'quora_qp', 'snli']
# train_raw = mz.datasets.wiki_qa.load_data("train")
# dev_raw = mz.datasets.wiki_qa.load_data("dev")
# test_raw = mz.datasets.wiki_qa.load_data("test")
# print(train_raw.left.head())
# print(train_raw.right.head())
# print(train_raw.relation.head())
# print(train_raw.frame().head())

# # # Slicing is extremely useful for partitioning data for training vs testing.
# # num_train = int(len(data_pack) * 0.8)
# # data_pack.shuffle(inplace=True)
# # train_slice = data_pack[:num_train]
# # test_slice = data_pack[num_train:]

# ### 数据预处理，BasicPreprocessor为指定预处理的方式，在预处理中包含了两步：fit,transform
# ### fit将收集一些有用的信息到preprocessor.context中，不会对输入DataPack进行处理
# ### transformer 不会改变context、DataPack,他将重新生成转变后的DataPack.
# ### 在transformer过程中，包含了Tokenize => Lowercase => PuncRemoval等过程，这个过程在方法中应该是可以自定义的
# print(mz.preprocessors.list_available())
# preprocessor = mz.preprocessors.BasicPreprocessor()
# preprocessor.fit(train_raw)  ## init preprocessor inner state.
# train_processed = preprocessor.transform(train_raw)
# test_processed = preprocessor.transform(test_raw)
# print(preprocessor.context)
# print(train_processed.left.head())

# vocab_unit = preprocessor.context["vocab_unit"]
# sequence = train_processed.left.loc["Q1"]["text_left"]
# print("Transformed Indices:", sequence)
# print(
#     "Transformed Indices Meaning:",
#     "_".join([vocab_unit.state["index_term"][i] for i in sequence]),
# )

# ### 创建模型以及修改参数（可以使用mz.models.list_available()查看可用的模型列表）
# print(mz.models.list_available())
# model = mz.models.DenseBaseline()
# print(model.params)
# model.params["task"] = task
# model.params["mlp_num_units"] = 3
# model.params.update(preprocessor.context)   # preprocessor.context['input_shapes']
# model.params.completed()
# # build and compile the model
# model.build()
# model.compile()
# model.backend.summary()

# # 超参数优化 hyperopt
# tuner = mz.auto.Tuner(
#     params=model.params,
#     train_data=train_processed,
#     test_data=test_processed,
#     num_runs=5
# )
# tuner.callbacks.append(mz.auto.tuner.callbacks.SaveModel())
# results = tuner.tune()
# print(results['best']['params'].to_frame())
# # best_model_id = results['best']['model_id']
# # mz.load_model(mz.USER_TUNED_MODELS_DIR.joinpath(best_model_id))


# ### 训练, 评估, 预测
# # x, y = train_processed.unpack()
# # test_x, test_y = test_processed.unpack()
# # model.fit(x, y, batch_size=32, epochs=5)
# # model.evaluate(test_x, test_y)
# # model.predict(test_x)

# # delaying expensive preprocessing steps or doing real-time data augmentation
# data_generator = mz.DataGenerator(train_processed, batch_size=32)
# test_generator = mz.DataGenerator(test_processed, batch_size=32)
# model.fit_generator(data_generator, epochs=5)   # , use_multiprocessing=True, workers=4
# model.evaluate_generator(test_generator)


# # ### 保存模型
# # model.save('my-model')
# # loaded_model = mz.load_model('my-model')

# exit()


"""
【文本匹配】MatchZoo的基本使用
http://element-ui.cn/article/show-29895.aspx

"""

# 1. 指定任务类型（一般情况可放在model之前，无论如何注意前后问题定义的一致性）
task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
# 1.1 定义任务的loss和metric
task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(task)     # Ranking Task

# 2. 准备数据
# 2.1 封装成DataPack
def build_data(path):
    data = {"sentence1":[], "sentence2":[], "label":[]}
    with codecs.open(path, "r", "utf-8") as fp:
        for line in fp:
            if not line:
                break
            if "#" in line:
                data['label'].append(int(line.strip().split(" ")[0]))
                comment = line[line.index("#")+1:].strip('\n').split(' ')
                data['sentence1'].append(re.sub("\'", "", comment[0]))
                data['sentence2'].append(re.sub("\'", "", comment[1]))
    return data

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


data_pack = read_data(setting.ltr_data_txt)
# data_pack.frame()

# 划分训练集和测试集
num_train = int(len(data_pack) * 0.8)
data_pack.shuffle(inplace=True)
train_pack_raw = data_pack[:num_train]
test_pack_raw = data_pack[num_train:]

# 2.2 经过preprocess处理
preprocessor = mz.models.MVLSTM.get_default_preprocessor()
preprocessor._units = [
    mz.preprocessors.units.tokenize.ChineseTokenize(),  # 采用结巴分词
    mz.preprocessors.units.punc_removal.PuncRemoval(),
]
# preprocessor = mz.preprocessors.chinese_preprocessor.ChinesePreprocessor()
print(preprocessor)
preprocessor.fit(train_pack_raw)  ## init preprocessor inner state.
print(preprocessor.context)
train_processed = preprocessor.transform(train_pack_raw)
test_processed = preprocessor.transform(test_pack_raw)

# # 加载embedding,设置文件路径
# path_vec = "D:/wordEmbedding/Tencent_AILab_ChineseEmbedding_small.txt"
# emb = mz.embedding.load_from_file(path_vec, mode="word2vec")
# # glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
# term_index = preprocessor.context['vocab_unit'].state['term_index']
# embedding_matrix = emb.build_matrix(term_index)
# l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
# embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

#　2.3 Dataset
trainset = mz.DataGenerator(
    data_pack=train_processed,
    mode='pair',
    num_dup=1,
    num_neg=4,
    batch_size=32
)
validset = mz.dataloader.Dataset(
    data_pack=test_processed,
    mode='point',
    batch_size=32
)

# 2.4 DataLoader 
padding_callback = mz.models.DRMM.get_default_padding_callback()    # padding策略

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)

validloader = mz.dataloader.DataLoader(
    dataset=validset,
    stage='dev',
    callback=padding_callback
)

# 3. 搭建模型
# 3.1 调用内置模型并初始化参数
model = mz.models.MVLSTM()
model.params['task'] = task
# model.params['embedding'] = embedding_matrix
model.params['embedding_output_dim'] = 200
model.params['embedding_input_dim'] = preprocessor.context['embedding_input_dim']
model.params["mlp_num_layers"] = 1
model.params["mlp_num_units"] = 10
model.params["mlp_num_fan_out"] = 1
model.params["mlp_activation_func"] = "tanh"
model.guess_and_fill_missing_params(verbose=0)
model.params.completed()
model.build()
model.backend.summary()
model.compile()
print(model)
print(
    "Trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad)
)

# 3.3 封装训练器
trainer = mz.trainers.Trainer(
    model=model,
    optimizer='adam',
    trainloader=trainloader,
    validloader=validloader,
    epochs=10
)

# 3.4 进行训练
trainer.run()

# 3.4 训练评估
train_generator = mz.PairDataGenerator(
    train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True
)
valid_x, valid_y = test_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(
    model, x=valid_x, y=valid_y, batch_size=len(valid_x)
)
history = model.fit_generator(
    train_generator,
    epochs=20,
    callbacks=[evaluate],
    workers=5,
    use_multiprocessing=False,
)

# 4. 进行预测
# trainer.predict()