import os
import sys
import zipfile
import codecs
import csv
from tqdm import tqdm
from bert_serving.client import BertClient
# from sentence_transformers import SentenceTransformer
import threading
import helpers
import setting

sys.path.append(r"../../")
import utils_toolkit as ut

"""sbert主问题向量预生成脚本
python bert_embedding_util.py -yasuo

bert_as_service 对并发的支持不太友好，需要加锁使用！
https://github.com/hanxiao/bert-as-service#starting-bertserver-from-python
https://github.com/hanxiao/bert-as-service#broadcasting-to-multiple-clients
https://github.com/hanxiao/bert-as-service#q-i-encounter-zmqerrorzmqerror-operation-cannot-be-accomplished-in-current-state-when-using-bertclient-what-should-i-do
"""

# 读取配置文件
DB_CONFIG, SBERT_CONFIG = helpers.get_config_info()
stop_words = ut.StringHelper.get_stopword("data/中文停用词表.txt")  # 停用词


def yasuo():
    """预生成主问题的bert向量表示（生产环境），并压缩保存
    每次更新知识库主问题后，需重新生成一次!
    """
    # 读取生产库的主问题数据
    primary_questions = []
    if os.path.exists(setting.primary_sim_file):
        with codecs.open(setting.primary_sim_file, "r", "utf-8") as csvFile:
            reader = csv.reader(csvFile)
            for item in reader:
                # 忽略第一行标题
                if reader.line_num == 1:
                    continue
                primary_questions.append(item[0])
                primary_questions.extend(item[1].split('###'))
    
    primary_questions = list(map(lambda q: helpers.clean(q, stop_words), primary_questions))
    primary_questions = list(set(primary_questions))    # 去重
    primary_questions = list(filter(None, primary_questions))   # 去除空字符串
    
    # 获取sbert向量
    sen2bertvec, flag = helpers.get_sbertvec(primary_questions, SBERT_CONFIG["url"])
    # 若有新增主问题，则重写入sbert向量文件
    if not flag:
        print("写入sbert向量文件...")
        with codecs.open(setting.bert_txt_file, "w", "utf-8") as f:
            for sent in primary_questions:
                sent_vec = list(sen2bertvec[sent])
                assert len(sent_vec)==768
                sent_vec = " ".join(map(lambda x: str(round(x, 6)), sent_vec))
                f.write("{}\t{}".format(sent, sent_vec))
                f.write("\n")

    # #Store sentences & embeddings on disc
    # import pickle
    # with open('embeddings.pkl', "wb") as fOut:
    #     pickle.dump(sen2bertvec, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # 压缩
    zip_file = zipfile.ZipFile(setting.bert_zip_file, 'w')
    # 把文件压缩成一个压缩文件
    zip_file.write(setting.bert_txt_file, compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()
    print("压缩完成！")


def jieya():
    # 解压&覆盖
    zip_file = zipfile.ZipFile(setting.bert_zip_file)
    zip_extract = zip_file.extractall("./")
    zip_file.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请输入参数后运行！")
        sys.exit(0)
    
    if sys.argv[1] == "-yasuo":
        yasuo()
    elif sys.argv[1] == "-jieya":
        jieya()
    else:
        print("参数有误！")