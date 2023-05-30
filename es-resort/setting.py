# -*- coding: utf-8 -*-
import sys
import socket

"""用来作变量和常量的初始化"""


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


# 钉钉告警
url = "https://oapi.dingtalk.com/robot/send?access_token=b76ce9429362c7697d34db88d32176ff1b6682286362af82c53fa18394d91ba5"

# 根据脚本参数，来决定用那个环境配置
ip = get_host_ip() 
if ip.startswith("10.231.9") or ip.startswith("10.231.25"):  # pro
    cfg_file = "configs/config_online.ini"
    mode = 'pro'
elif ip.startswith("10.231.198") or ip.startswith("10.231.199"):  # pre
    cfg_file = "configs/config_pre.ini"
    mode = 'pre'
else:   # test
    cfg_file = "configs/config.ini"
    mode = 'test'

model_bucketA = 'bucketA/model/'
data_bucketA = 'bucketA/data/'
model_bucketB = 'bucketB/model/'
data_bucketB = 'bucketB/data/'

# 特征处理过程-中间对象保存
tfidf_vec_file = "tfidf_vec.job"
bow_vec_file = "bow_vec.job"
bm25_file = "bm25.job"
doc2vec_file = "saved_doc2vec_model"
sen2docvec_dict_file = "sen2docvec_dict"
time_feats_file = "time_feats.job"
ctr_file = "ctr.job"
cat_feat_file = "cat_feat.job"
stdsc_file = "stdsc.job"
label_encoder_file = "label_encoder.job"
# 特征处理结果文件保存
ltr_data_excel = "ltr_data.xlsx"
ltr_output_excel = "ltr_output.xlsx"
ltr_data_txt = "ltr_data.txt"
data_csv = "data_pre.csv"
# 划分特征集和group集
ltr_train = "train/raw_train.txt"
lxh_train = "train/feats.txt"
lxh_train_group = "train/group.txt"
ltr_valid = "dev/raw_dev.txt"
lxh_valid = "dev/feats.txt"
lxh_valid_group = "dev/group.txt"
ltr_test = "test/raw_test.txt"
lxh_test = "test/feats.txt"
lxh_test_group = "test/group.txt"
# 汇总所有使用中的特征
feature_all_file = "feature_name_all.txt"

# GBDT模型相关
gbdt_model = "{}.mod"
best_iteration = "{}.best_iteration.txt"
best_trial = "best_trial.txt"  # 最佳trial_number
feature_txt = "feature_importance.txt"

# log文件的全路径
log_file = "logs/flask_request8099.log"
# 定义日志输出格式 ↓
STDOUT_LOG_FMT = "%(log_color)s[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
STDOUT_DATE_FMT = "%Y-%m-%d %H:%M:%S"
FILE_LOG_FMT = "[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d][%(thread)d] - %(message)s"
FILE_DATE_FMT = "%Y-%m-%d %H:%M:%S"
# 定义日志输出格式 ↑

# save_path = 'model/my-model'
# preprocessor_path = 'model/p'

# sbert相关
bert_zip_file = 'sen2bertvec.zip'
primary_file = "data/primary.csv"
primary_sim_file = "data/primary_sim.csv"
bert_txt_file = "data/sen2bertvec.txt"
# sbert_path = "../../distilbert-multilingual-nli-stsb-quora-ranking"

stopword_file = "data/中文停用词表.txt"